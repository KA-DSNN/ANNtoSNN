# Notes
## Poisson inputs

links: 
* https://brian2.readthedocs.io/en/stable/user/input.html#poisson-inputs
* https://brian2.readthedocs.io/en/stable/user/input.html#more-on-poisson-inputs

If the given value for rates is a constant, then using ```PoissonGroup(N, rates)``` is equivalent to:

``` python
NeuronGroup(N, 'rates : Hz', threshold='rand()<rates*dt')
```

It means this neuron group generate certain number of spike in the mentioned dt
