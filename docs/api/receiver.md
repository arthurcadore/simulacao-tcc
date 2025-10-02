# Reception Chain

The reception chain is divided into two steps, carrier detection, responsible for verifying the received signal and identifying for each time segment the carriers present, and the receiver chain, responsible for receiving a signal segment and performing the reception process to recover the data vector $u_t^{(0)}$, as shown in the block diagram below.

![pageplot](../assets/reception_chain_hard.svg)
<!-- ![pageplot](../assets/reception_chain_soft.svg) -->

## Carrier Detector

::: detector.CarrierDetector.__init__
    options:
        extra:
            show_docstring: true
            show_signature: true

## Reception Chain

::: receiver.Receiver
    options:
        extra:
            show_docstring: true
            show_signature: true