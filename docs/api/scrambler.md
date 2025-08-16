# Scrambler

<p align="center" class="image-container">
  <img src="../assets/embaralhador.png" alt="Diagrama de blocos do embaralhador" class="responsive-image">
</p>

O processo de embaralhamento pode ser expresso como:

$$
X_n =
\begin{cases}
A, & \text{se } n \equiv 0 \pmod{3} \\\\
B, & \text{se } n \equiv 1 \pmod{3} \\\\
C, & \text{se } n \equiv 2 \pmod{3}
\end{cases}, \quad
Y_n =
\begin{cases}
A, & \text{se } n \equiv 0 \pmod{3} \\\\
B, & \text{se } n \equiv 1 \pmod{3} \\\\
C, & \text{se } n \equiv 2 \pmod{3}
\end{cases}
$$

::: scrambler.Scrambler
    options:
        extra:
            show_docstring: true
            show_signature: true
