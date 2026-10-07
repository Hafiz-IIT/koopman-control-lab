from koopman_lift import lift, linear_step

x=0.7
z=lift(x)
z_next=linear_step(z)
print("Koopman-inspired lift demo")
print("state:",x)
print("lifted:",z)
print("next lifted state:",z_next)
