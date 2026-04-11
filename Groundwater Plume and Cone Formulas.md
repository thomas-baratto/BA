# Groundwater Plume and Cone Formulas

Foundational mathematical formulas and general parameters for plotting the boundaries of groundwater thermal plumes and hydraulic depression cones.

## 1. Plotting the Depression Cone

To generate a mathematically accurate depression cone, calculate the drawdown ($s$) at various radial distances ($r$) from the extraction well.

### The Theis Equation (Transient Drawdown)

The standard formula used to plot the drawdown curve of a depression cone over a specific pumping duration:

$$s = \frac{Q}{4 \pi T} W(u)$$

Where the argument $u$ is defined as:

$$u = \frac{r^2 S}{4 T t}$$

- $s$ = drawdown (the vertical drop in the water table)
- $Q$ = constant pumping rate of the well
- $T$ = transmissivity of the aquifer
- $S$ = storativity of the aquifer
- $t$ = time since pumping began
- $W(u)$ = the well function, typically expanded as an infinite series:
  $W(u) = -0.5772 - \ln(u) + u - \frac{u^2}{2 \cdot 2!} + \frac{u^3}{3 \cdot 3!} - \dots$

### Radius of Influence (Sichardt's Formula)

To plot the absolute outer boundary of the depression cone (where drawdown reaches zero) in an unconfined aquifer, use Sichardt's empirical formula:

$$R = 3000 \cdot s \cdot \sqrt{K}$$

- $R$ = radius of influence in meters
- $s$ = maximum drawdown at the well in meters
- $K$ = hydraulic conductivity in meters per second

## 2. Plotting the Thermal Plume

Analytical solutions can map the spatial metrics (width and distance) of the thermal plume, assuming a homogeneous aquifer.

### Maximum Plume Width (Iso-width)

According to the analytical models presented by Piga et al. (2017) and Banks (2011), the asymptotic plume width transverse to the groundwater flow direction can be calculated as:

$$y = \frac{Q_{pl}}{b \cdot v_D}$$

- $y$ = the asymptotic plume width (iso-width)
- $Q_{pl}$ = the fraction of the injected flow rate that escapes down-gradient (not captured by an extraction well)
- $b$ = the physical thickness of the aquifer
- $v_D$ = the Darcy velocity (or Darcy flux) of the background groundwater

### Plume Length (Iso-distance)

The longitudinal distance the thermal plume travels downstream over a given operational time can be approximated using the thermal advective velocity:

$$x = v_{th} \cdot t$$

Where the thermal advective velocity ($v_{th}$) is calculated as:

$$v_{th} = \frac{v_D}{n_e \cdot R_{th}}$$

- $x$ = the downstream iso-distance from the injection source
- $t$ = operational time
- $n_e$ = the effective porosity of the porous medium
- $R_{th}$ = the thermal retardation factor

### Thermal Retardation Factor ($R_{th}$)

Heat moves slower than the physical groundwater due to the aquifer matrix absorbing thermal energy. To accurately plot the spatial lag of the thermal front, apply the thermal retardation factor:

$$R_{th} = \frac{\rho_b c_b}{n_e \rho_f c_f}$$

- $\rho_b c_b$ = the volumetric heat capacity of the bulk aquifer material (solid matrix plus fluid)
- $\rho_f c_f$ = the volumetric heat capacity of the groundwater fluid itself

## 3. General Tips for Thesis Visualizations

- **Plotting the Cone:** Plot the distance ($r$) on a logarithmic x-axis and the drawdown ($s$) on a linear y-axis. This semi-logarithmic setup yields the classic straight-line intercept that makes it easy to visually identify the radius of influence.
- **Overlaying the Plots:** To visualize the interaction between the extraction mechanics and the heat transport, treat the extraction area as a sink and plot the hydraulic head contours underneath the thermal plume. The thermal plume boundary (e.g., the $\Delta T = 1°\text{C}$ isotherm) can be mapped dynamically across these flow paths to show how the cone alters the plume's trajectory.