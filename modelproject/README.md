# Model analysis project

Our project is titled **The Solow Model** and is about a macroeconomic model that investigates the sources of economic growth in an economy.

The solow model assumes that the economy is a single company representative of the entire economy.
The model is based upon a fixed labor force, a production function that yields constant returns to scale and diminishing returns to capital accumulatio as seen in previous macroecnomics courses.

Furthermore, the economy is also closed, i.e there are no imports or exports. The variables used in the model are Y (output), K (capital stock), L (labor force), A (technological progress or total factor productivity), s (savings rate), δ (depreciation rate or rate at which capital stock depreciates), n (labor force growth rate), a (output elasticity of capital), and 1-a (output elasticity of labor). As we known them from priveous macroeconomics courses.


The reuslts of our project can be seen from running our project "modelproject.ipynb".
 Our code gives two ways of solving the solow model, an analytical solution using the sympy tool, in order to derive the steady-state values of capital and output per capital
 Furtermore a numerical solution using the scipy.optimize tool to find the steady-state values by maximizing the output per capita equation. The numerical solution tests for convergence by providing five different initial values of capital per capita.

 Lastly we plot the convergence to the steady state using different growth rates in the technological factor.