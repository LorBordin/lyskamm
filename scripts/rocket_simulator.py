import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class EngineParser:
    """Parser per file .eng (formato RASP)"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.name = ""
        self.diameter = 0
        self.length = 0
        self.propellant_mass = 0
        self.total_mass = 0
        self.time = []
        self.thrust = []
        
    def parse(self):
        """Legge e analizza il file .eng"""
        encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(self.filepath, 'r', encoding=encoding) as f:
                    lines = f.readlines()
            except (UnicodeDecodeError, UnicodeError):
                continue  # prova il prossimo encoding
        
        # Salta commenti e trova la riga di specifiche
        data_start = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            # Riga di specifiche (es: F35 29 95 0-4-6-8 0.090 0.118 TSP)
            parts = line.split()
            if len(parts) >= 6:
                self.name = parts[0]
                self.diameter = float(parts[1])  # mm
                self.length = float(parts[2])    # mm
                # parts[3] sono i delay disponibili
                self.propellant_mass = float(parts[4])  # kg
                self.total_mass = float(parts[5])       # kg
                data_start = i + 1
                break
        
        # Leggi i dati di spinta (tempo, forza)
        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    time = float(parts[0])
                    thrust = float(parts[1])
                    self.time.append(time)
                    self.thrust.append(thrust)
                except ValueError:
                    continue
        
        self.time = np.array(self.time)
        self.thrust = np.array(self.thrust)
        
        return self
    
    def get_burn_time(self):
        """Restituisce la durata totale della combustione"""
        return self.time[-1] if len(self.time) > 0 else 0
    
    def get_average_thrust(self):
        """Calcola la spinta media"""
        if len(self.time) < 2:
            return 0
        # Integrazione trapezoidale
        total_impulse = np.trapezoid(self.thrust, self.time)
        return total_impulse / self.get_burn_time()
    
    def get_thrust_interpolator(self):
        """Crea una funzione di interpolazione per la spinta"""
        if len(self.time) < 2:
            return lambda t: 0
        return interp1d(self.time, self.thrust, kind='linear', 
                       bounds_error=False, fill_value=0)


class RocketSimulator:
    def __init__(self, total_mass, diameter, cd, engine=None, 
                 thrust=None, burn_time=None, propellant_mass=None):
        """
        Parametri:
        - total_mass: peso del razzo completo con motore (kg)
        - diameter: diametro frontale del razzo (m)
        - cd: coefficiente di attrito (drag coefficient)
        
        Opzione 1 - Usa file .eng:
        - engine: oggetto EngineParser con dati del motore
        
        Opzione 2 - Usa parametri manuali:
        - thrust: forza media del motore (N)
        - burn_time: durata del motore (s)
        - propellant_mass: peso del propellente (kg)
        """
        self.diameter = diameter
        self.cd = cd
        self.area = np.pi * (diameter / 2) ** 2
        
        # Costanti
        self.g = 9.81
        self.rho = 1.225
        
        # Configura motore
        if engine is not None:
            # Usa dati dal file .eng
            self.engine_mode = 'curve'
            self.thrust_func = engine.get_thrust_interpolator()
            self.burn_time = engine.get_burn_time()
            self.mp = engine.propellant_mass
            self.m0 = total_mass
            self.mf = total_mass - engine.propellant_mass
            self.engine_name = engine.name
            self.avg_thrust = engine.get_average_thrust()
        else:
            # Usa parametri manuali
            self.engine_mode = 'constant'
            self.thrust_value = thrust
            self.burn_time = burn_time
            self.mp = propellant_mass
            self.m0 = total_mass
            self.mf = total_mass - propellant_mass
            self.engine_name = "Custom"
            self.avg_thrust = thrust
    
    def thrust(self, t):
        """Restituisce la spinta al tempo t"""
        if t > self.burn_time:
            return 0
        
        if self.engine_mode == 'curve':
            return self.thrust_func(t)
        else:
            return self.thrust_value
    
    def drag_force(self, velocity, altitude):
        """Calcola la forza di attrito aerodinamico"""
        rho_alt = self.rho * np.exp(-altitude / 8500)
        return 0.5 * rho_alt * velocity**2 * self.cd * self.area
    
    def mass(self, t):
        """Calcola la massa istantanea del razzo"""
        if t <= self.burn_time:
            return self.m0 - (self.mp / self.burn_time) * t
        else:
            return self.mf
    
    def simulate(self, dt=0.001):
        """Simula il volo del razzo con passo temporale più piccolo per precisione"""
        t = 0
        h = 0
        v = 0
        
        times = [t]
        heights = [h]
        velocities = [v]
        thrusts = [self.thrust(t)]
        
        v_at_2m = None
        t_at_2m = None
        
        # Fase ascendente
        while True:
            m = self.mass(t)
            
            thrust_force = self.thrust(t)
            weight = m * self.g
            drag = self.drag_force(v, h) if v > 0 else 0
            
            a = (thrust_force - weight - drag) / m
            
            v += a * dt
            h += v * dt
            t += dt
            
            times.append(t)
            heights.append(h)
            velocities.append(v)
            thrusts.append(thrust_force)
            
            if v_at_2m is None and h >= 2.0:
                v_at_2m = v
                t_at_2m = t
            
            if v <= 0 and h > 0:
                apogee = h
                time_to_apogee = t
                break
            
            if t > 300 or h < -10:
                break
        
        return {
            'apogeo': apogee,
            'tempo_apogeo': time_to_apogee,
            'velocita_2m': v_at_2m if v_at_2m else 0,
            'tempo_2m': t_at_2m if t_at_2m else 0,
            'times': times,
            'heights': heights,
            'velocities': velocities,
            'thrusts': thrusts
        }
    
    def print_results(self, results):
        """Stampa i risultati in modo formattato"""
        print("=" * 60)
        print("RISULTATI SIMULAZIONE RAZZO")
        print("=" * 60)
        print(f"Motore:                    {self.engine_name}")
        print(f"Spinta media:              {self.avg_thrust:.2f} N")
        print(f"Durata combustione:        {self.burn_time:.3f} s")
        print(f"Massa propellente:         {self.mp:.3f} kg")
        print(f"Massa iniziale/finale:     {self.m0:.3f} / {self.mf:.3f} kg")
        print("-" * 60)
        print(f"Apogeo:                    {results['apogeo']:.2f} m")
        print(f"Tempo all'apogeo:          {results['tempo_apogeo']:.2f} s")
        print(f"Velocità a 2m (rampa):     {results['velocita_2m']:.2f} m/s")
        print(f"Tempo a 2m:                {results['tempo_2m']:.3f} s")
        print("=" * 60)
    
    def plot_results(self, results):
        """Crea grafici dei risultati"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
        
        # Grafico spinta vs tempo
        ax1.plot(results['times'], results['thrusts'], 'orange', linewidth=2)
        ax1.axvline(x=self.burn_time, color='r', linestyle='--', 
                    label=f"Fine combustione ({self.burn_time:.2f}s)")
        ax1.set_xlabel('Tempo (s)')
        ax1.set_ylabel('Spinta (N)')
        ax1.set_title(f'Curva di Spinta - {self.engine_name}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Grafico altezza vs tempo
        ax2.plot(results['times'], results['heights'], 'b-', linewidth=2)
        ax2.axhline(y=results['apogeo'], color='r', linestyle='--', 
                    label=f"Apogeo: {results['apogeo']:.2f} m")
        ax2.axhline(y=2, color='g', linestyle='--', alpha=0.5, 
                    label="Uscita rampa (2m)")
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('Altezza (m)')
        ax2.set_title('Altezza vs Tempo')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Grafico velocità vs tempo
        ax3.plot(results['times'], results['velocities'], 'r-', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.axvline(x=self.burn_time, color='orange', linestyle='--', 
                    label=f"Fine combustione ({self.burn_time:.2f}s)")
        ax3.set_xlabel('Tempo (s)')
        ax3.set_ylabel('Velocità (m/s)')
        ax3.set_title('Velocità vs Tempo')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # CONFIGURAZIONE - Scegli una delle due opzioni
    # ========================================================================
    
    # OPZIONE 1: Usa file .eng (lascia None per usare parametri manuali)
    engine_file = './motor_data/TSP_F35.eng'  # Percorso al file .eng, oppure None
    
    # OPZIONE 2: Parametri manuali (usati solo se engine_file è None)
    manual_thrust = 105.0          # Forza media del motore (N)
    manual_burn_time = 115/105      # Durata motore (s)
    manual_propellant_mass = 0.130 # Peso propellente (kg)
    
    # Parametri del razzo (sempre necessari)
    rocket_mass_empty = 0.910  # Massa del razzo vuoto senza motore (kg)
    diameter = 0.060           # Diametro frontale (m)
    cd = 0.9                  # Coefficiente di attrito
    
    # ========================================================================
    # SIMULAZIONE
    # ========================================================================
    
    if engine_file is not None:
        # Usa file .eng
        print("SIMULAZIONE CON FILE .ENG")
        print(f"Caricamento motore da: {engine_file}")
        print()
        
        try:
            engine = EngineParser(engine_file).parse()
            total_mass = rocket_mass_empty + engine.propellant_mass
            
            sim = RocketSimulator(total_mass, diameter, cd, engine=engine)
            
        except FileNotFoundError:
            print(f"ERRORE: File '{engine_file}' non trovato!")
            print("Usa parametri manuali oppure verifica il percorso del file.")
            exit(1)
        except Exception as e:
            print(f"ERRORE nel parsing del file .eng: {e}")
            exit(1)
    
    else:
        # Usa parametri manuali
        print("SIMULAZIONE CON PARAMETRI MANUALI")
        print()
        
        total_mass = rocket_mass_empty + manual_propellant_mass
        
        sim = RocketSimulator(total_mass, diameter, cd,
                             thrust=manual_thrust,
                             burn_time=manual_burn_time,
                             propellant_mass=manual_propellant_mass)
    
    # Esegui la simulazione
    results = sim.simulate(dt=0.001)
    
    # Stampa i risultati
    sim.print_results(results)
    
    # Mostra i grafici
    sim.plot_results(results)