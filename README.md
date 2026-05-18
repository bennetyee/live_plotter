# Live Plotter

The code in this repo is vibe coded.

Usage:

```
$ get_circuit_info  -a --abs -k instantPowerW --live 5 | eval live_plotter --labels $(get_circuit_info -a -k name -q)
```

See https://github.com/bennetyee/SPAN-hacks for `get_circuit_info`.
