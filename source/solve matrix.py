# Re-import necessary functions and redefine symbols
from sympy import symbols, solve

# Redefine all symbols since the context was lost
E, gamma_e, B_x, B_y, B_z, lambda_ = symbols('E gamma_e B_x B_y B_z lambda_')
a = -4  # Coefficient of E^2 in both equations
eq1_b = -8 * B_z  # Coefficient of E in the first equation
eq2_b = 8 * B_z   # Coefficient of E in the second equation
c_common = 4*B_x**2*gamma_e**2 + 4*B_y**2*gamma_e**2 + 4*B_z**2*gamma_e**2 - 4*B_z**2 + lambda_**2

# Recalculate c1 and c2 with the corrected variable names and definitions
c1 = c_common + 4*B_z*gamma_e*lambda_
c2 = c_common - 4*B_z*gamma_e*lambda_

# Solve the quadratic equations again with corrected definitions
E1_solutions = solve(a*E**2 + eq1_b*E + c1, E)
E2_solutions = solve(a*E**2 + eq2_b*E + c2, E)

print(E1_solutions, E2_solutions)