// F35/G115 flight check with OpenRocket 24.12 (headless).
// Run: java -Djava.awt.headless=true -cp ~/OpenRocket/jar/OpenRocket-24.12.jar scripts/openrocket/Sim.java lyskamm.ork scripts/motor_data/TSP_F35.eng [rod_m=2.0] [adapter_g=0] [delays: plugged | 4,6,8]
// Parachute deployment follows the .ork (currently: at apogee, by the altimeter).
// Uses system Java (>=17) source launcher. Adds an extra mass in the lower e-bay tube to bracket the unknown masses.
import info.openrocket.core.file.GeneralRocketLoader;
import info.openrocket.core.document.*;
import info.openrocket.core.startup.Application;
import info.openrocket.core.plugin.PluginModule;
import info.openrocket.swing.startup.GuiModule;
import info.openrocket.core.file.motor.RASPMotorLoader;
import info.openrocket.core.motor.*;
import info.openrocket.core.rocketcomponent.*;
import info.openrocket.core.simulation.*;
import com.google.inject.*;
import java.io.*;
import java.util.*;
public class Sim {
  public static void main(String[] a) throws Exception {
    GuiModule g = new GuiModule();
    Application.setInjector(Guice.createInjector(g, new PluginModule())); g.startLoader();
    String ork=a[0], eng=a[1];
    var builders = new RASPMotorLoader().load(new FileInputStream(eng), eng);
    Motor motor = builders.get(0).build();
    System.out.printf("motor %s Ø%.0f len %.0f total impulse %.1f Ns, launch mass %.0f g%n", motor.getDesignation(), motor.getDiameter()*1000, motor.getLength()*1000, motor.getTotalImpulseEstimate(), motor.getLaunchMass()*1000);
    double[] extra = {0, 150, 200, 250};
    String[] delays = (a.length>4 ? a[4] : "plugged").split(","); // "plugged" = no ejection charge (altimeter deployment)
    double rod = a.length>2 ? Double.parseDouble(a[2]) : 2.0; double adapter = a.length>3 ? Double.parseDouble(a[3]) : 0.0;
    for (double ex : extra) for (String dl : delays) {
      OpenRocketDocument d = new GeneralRocketLoader(new File(ork)).load();
      Rocket r = d.getRocket();
      InnerTube mount=null; BodyTube ebay=null;
      for (RocketComponent c : r) { if (c instanceof InnerTube) mount=(InnerTube)c; if ("E-bay tube (lower)".equals(c.getName())) ebay=(BodyTube)c; }
      if (ex>0) { MassComponent m = new MassComponent(); m.setName("missing mass"); m.setComponentMass(ex/1000); m.setLength(0.05); ebay.addChild(m); }
      if (adapter>0) { MassComponent ad = new MassComponent(); ad.setName("F35 adapter"); ad.setComponentMass(adapter/1000); ad.setLength(0.095); ad.setAxialMethod(info.openrocket.core.rocketcomponent.position.AxialMethod.BOTTOM); ad.setAxialOffset(0.0415); mount.addChild(ad); }
      FlightConfigurationId fcid = new FlightConfigurationId(); FlightConfiguration fc = new FlightConfiguration(r, fcid); r.setFlightConfiguration(fcid, fc); r.setSelectedConfiguration(fcid);
      MotorConfiguration mc = new MotorConfiguration(mount, fc.getId());
      mc.setMotor(motor); mc.setEjectionDelay(dl.equals("plugged") ? Motor.PLUGGED_DELAY : Double.parseDouble(dl));
      mount.setMotorMount(true); mount.setMotorConfig(mc, fc.getId());
      fc.update();
      Simulation sim = new Simulation(d, r);
      sim.setFlightConfigurationId(fc.getId());
      sim.getOptions().setLaunchRodLength(rod);
      sim.simulate();
      FlightData fd = sim.getSimulatedData();
      var br = fd.getBranch(0);
      List<Double> t = br.get(FlightDataType.TYPE_TIME), st = br.get(FlightDataType.TYPE_STABILITY), v=br.get(FlightDataType.TYPE_VELOCITY_TOTAL), h=br.get(FlightDataType.TYPE_ALTITUDE), mass=br.get(FlightDataType.TYPE_MASS);
      double stRod=Double.NaN, stMin=99;
      // stability at first time v >= rod velocity, and min during motor burn
      for (int i=0;i<t.size();i++){ Double s=st.get(i); if (s==null||s.isNaN()) continue; if (Double.isNaN(stRod) && v.get(i)>=fd.getLaunchRodVelocity()*0.999) stRod=s; if (t.get(i)<motor.getBurnTimeEstimate() && v.get(i)>5) stMin=Math.min(stMin,s); }
      System.out.printf("extra %3.0f g | liftoff %4.0f g | delay %s | rod-exit %4.1f m/s | stab@rod %.2f cal, min(burn) %.2f | apogee %5.1f m @ %4.1f s | vmax %5.1f m/s | v@deploy %5.1f m/s | descent %4.1f m/s | optimum delay %4.1f s | warnings %s%n",
        ex, mass.get(0)*1000, dl, fd.getLaunchRodVelocity(), stRod, stMin, fd.getMaxAltitude(), fd.getTimeToApogee(), fd.getMaxVelocity(), fd.getDeploymentVelocity(), fd.getGroundHitVelocity(), br.getOptimumDelay(), fd.getWarningSet());
    }
    System.exit(0);
  }
}
