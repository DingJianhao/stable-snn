# stable-snn
Code for 'Robust Stable Spiking Neural Networks' in ICML2024

# Summary

Spiking neural networks (SNNs) are gaining popularity in deep learning due to their low energy budget on neuromorphic hardware. However, they still face challenges in lacking sufficient robustness to guard safety-critical applications such as autonomous driving. Many studies have been conducted to defend SNNs from the threat of adversarial attacks. This paper aims to uncover the robustness of SNN through the lens of the stability of nonlinear systems. We are inspired by the fact that searching for parameters altering the leaky integrate-and-fire dynamics can enhance their robustness. Thus, we dive into the dynamics of membrane potential perturbation and simplify the formulation of the dynamics. We present that membrane potential perturbation dynamics can reliably convey the intensity of perturbation. Our theoretical analyses imply that the simplified perturbation dynamics satisfy input-output stability. Thus, we propose a training framework with modified SNN neurons and to reduce the mean square of membrane potential perturbation aiming at enhancing the robustness of SNN. Finally, we experimentally verify the effectiveness of the framework in the setting of Gaussian noise training and adversarial training on the image classification task.
