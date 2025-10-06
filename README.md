# Lyskamm Rocket Project 🚀

**Lyskamm** is a mid-power rocket prototype that is entirely **3D-printed** and **assembled without glue**.  
Each component is designed as an independent part that can be easily replaced or upgraded at any time.  
Assembly is achieved through **mechanical joints and screws**, ensuring modularity and maintainability.

## Project Structure

``` 
lyskamm/
├── assemblies/ # Full rocket assembly (OpenRocket, FreeCAD files)
└──components/ # Individual parts (.FCStd, .STEP, .3MF)
```

## Components

Each component (e.g., fins, nose cone, fuselage, motor block) includes:
- `.FCStd` — Original FreeCAD design file  
- `.STEP` — 3D model for CAD interoperability  
- `.3MF` — File ready for 3D printing (optimized for BambuLab P1S)

## Assembly

The rocket is fully modular:
- **No adhesives required**  
- **Interlocking joints and screws** ensure secure connections  
- Components can be replaced individually for testing, iteration, or repair

## Future Development

Planned additions:
- Experimental motor design studies  
- Launch and test reports  
- Performance analysis and simulation data

## License

This project is released for educational and research purposes.  
A specific license will be defined in future updates.
