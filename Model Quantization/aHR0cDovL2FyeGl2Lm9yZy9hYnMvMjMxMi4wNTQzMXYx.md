# Efficient Quantization Strategies for Latent Diffusion Models
Yuewei Yang, Xiaoliang Dai, Jialiang Wang, Peizhao Zhang, Hongbo Zhang

## 🧩 Problem to Solve
Latent Diffusion Models (LDMs)는 텍스트-이미지 생성과 같은 다양한 애플리케이션에서 뛰어난 성능을 보이지만, 방대한 파라미터 수로 인해 엣지 디바이스에 배포하기 어렵다는 문제가 있습니다. 딥러닝 모델의 크기를 줄이는 Post Training Quantization (PTQ)은 이러한 제약을 해결하기 위한 유망한 방법이지만, LDM에 적용될 때는 다음과 같은 복잡성으로 인해 어려움을 겪습니다:
*   **시간적 및 구조적 복잡성**: LDM의 시변(temporal) 및 복잡한 UNet 구조가 양자화 오류를 증폭시킵니다.
*   **공간 압축**: 로컬 다운샘플링 및 업샘플링 과정에서 양자화 오류가 크게 확대될 수 있습니다.
*   **동적 활성화 아웃라이어**: 전역적(global)으로 시변하는 활성화 아웃라이어가 양자화 성능을 저해합니다.
기존 PTQ 방법들은 계산 복잡도를 크게 증가시키거나 원본 훈련 데이터 또는 모델 재훈련을 요구합니다. 따라서 LDM을 효율적으로 압축하면서도 성능 손실을 최소화하는 새로운 양자화 전략이 필요합니다.

## ✨ Key Contributions
*   **효율적인 양자화 전략 제안**: 상대적 양자화 노이즈 분석을 통해 LDM의 전역적(블록) 및 지역적(모듈) 수준에서 효과적인 양자화 솔루션을 결정하는 최초의 연구입니다.
*   **SQNR(Signal-to-Quantization-Noise Ratio) 지표 활용**: 누적된 전역 양자화 노이즈와 상대적 지역 양자화 노이즈를 모두 설명하기 위해 SQNR을 효율적인 지표로 채택했습니다.
*   **전역 하이브리드 양자화(Global Hybrid Quantization)**: 높은 상대적 양자화 노이즈를 완화하기 위해 특정 LDM 블록에 더 높은 정밀도의 양자화를 적용하는 하이브리드 접근 방식을 제안했습니다.
*   **지역 노이즈 보정(Local Noise Correction)**: 활성화 양자화 노이즈를 완화하기 위해 가장 민감한 모듈에 스무딩 메커니즘을 구현했습니다.
*   **단일 샘플링 스텝 캘리브레이션(Single-sampling-step Calibration)**: 순방향 확산 과정의 마지막 단계에서 확산 노이즈가 최고조에 달할 때 양자화에 대한 지역 모듈의 견고함을 활용하여, 캘리브레이션 효율성과 성능을 크게 향상시켰습니다.

## 📎 Related Works
*   **확산 모델 (Diffusion Models, DMs)**: Ho et al. [13], Sohl-Dickstein et al. [39] 등의 선행 연구를 기반으로 합니다.
*   **잠재 확산 모델 (Latent Diffusion Models, LDMs)**: Rombach et al. [34], Podell et al. [33] 등 텍스트 인코더와 VAE를 결합한 LDM의 성공을 언급합니다.
*   **효율적인 모델 배포 기술**: 가지치기(pruning) [9], 증류(distillation) [27, 36, 41] 등이 있지만, 본 연구는 UNet 구조 변경이나 재훈련이 필요 없는 PTQ에 집중합니다.
*   **기존 확산 모델 PTQ**: PTQ4DM [37], Q-Diffusion [21]은 재구성 기반 PTQ 방법을 사용하며, PTQD [11]는 양자화 노이즈를 확산 노이즈와 융합합니다.
*   **활성화 아웃라이어 해결**: SmoothQuant [48]는 대규모 언어 모델(LLMs)에서 활성화 아웃라이어 문제를 해결하기 위해 사용되었으며, 본 연구에서 민감한 모듈에 적용됩니다.
*   **양자화 관련 지표**: SQNR은 CNN 양자화 [18, 38] 및 확산 모델의 소프트맥스 [31]에서 양자화 손실을 평가하는 데 사용된 바 있습니다.

## 🛠️ Methodology
본 연구는 상대적 양자화 노이즈를 통해 LDM 양자화를 분석하고 효율적인 양자화 전략을 제안합니다.
1.  **상대적 양자화 노이즈 분석 (SQNR)**:
    *   전체 모델 ($\theta_{fp}$)과 양자화된 모델 ($\theta_{q}$)의 출력 차이를 양자화 노이즈로 간주합니다.
    *   **SQNR(Signal-to-Quantization-Noise Ratio)**을 핵심 지표로 사용합니다:
        $$ SQNR_{\xi,t} = 10\log \frac{E_{z}||\xi_{fp}(z_t)||_{2}^{2}}{||\xi_{q}(\hat{z}_t)-\xi_{fp}(z_t)||_{2}^{2}} $$
        여기서 $\xi$는 전체 모델 또는 특정 모듈을 나타냅니다.
    *   SQNR은 양자화 노이즈의 **누적성(accumulation)**, **상대적 비교(relative values)**, **효율적인 계산(convenient computation)**이라는 세 가지 주요 속성을 만족하여 민감한 블록/모듈 식별에 적합합니다. 특히, 적은 수의 샘플로 단일 순방향 패스 내에서 효율적으로 계산됩니다.
2.  **전역 하이브리드 양자화 전략**:
    *   균일한(homogeneous) 양자화는 낮은 비트 정밀도에서 실패합니다.
    *   $SQNR_{\theta}$ (시간 평균 SQNR)을 기반으로 각 블록의 민감도를 측정합니다.
    *   실험을 통해 UNet의 출력에 가까운 업샘플링 블록들이 양자화에 매우 민감하다는 것을 발견했습니다 (Figure 3a, 5a).
    *   이러한 민감한 블록들에는 더 높은 정밀도(예: fp16)의 양자화를 적용하여 전체적인 양자화 노이즈를 완화합니다.
3.  **지역 노이즈 보정 전략**:
    *   각 모듈의 $SQNR_{\xi}$를 평가하여 민감한 모듈을 식별합니다 (Figure 6).
    *   일관적으로 양자화에 민감한 세 가지 유형의 연산을 확인했습니다: a) 공간 샘플링 연산, b) 어텐션 트랜스포머 블록 후의 투영 레이어, c) 업 블록의 숏컷 연결.
    *   **활성화 아웃라이어 완화**: 이들 민감 모듈의 활성화 범위에서 나타나는 아웃라이어 문제를 해결하기 위해 SmoothQuant [48]를 적용합니다.
        $$ \hat{Z}\hat{W}= (Z\text{diag}(s)^{-1})(\text{diag}(s)W) $$
        $$ s_j=\frac{\text{max}|Z_j|^\alpha}{\text{max}|W_j|^{1-\alpha}} $$
        여기서 $s_j$는 아웃라이어를 완화하고 양자화 부담을 가중치로 이동시키는 채널별 스케일입니다. 전체 모듈의 10% 정도에 해당하는 가장 민감한 모듈에 SmoothQuant를 적용하여 계산 오버헤드와 양자화 노이즈 감소 사이의 균형을 맞춥니다.
    *   **단일 샘플링 스텝 캘리브레이션**: 확산 노이즈가 가장 강한 순방향 확산 과정의 마지막 단계(첫 번째 추론 단계)에서만 양자화 파라미터를 캘리브레이션합니다. 이는 활성화 범위의 동적인 변화를 줄여 더 효율적이고 정확한 양자화를 가능하게 합니다.

## 📊 Results
*   **전역 하이브리드 양자화**: LDM 1.5, 2.1, XL 모델에서 8W8A, 4W8A 설정으로 평가했을 때, 모델의 운영 크기(Size)와 계산량(TBOPs)을 크게 줄이면서도 (예: LDM XL의 경우 약 75% 크기 감소) $SQNR_{\theta}$가 현저히 증가하여 양자화 노이즈가 감소했음을 입증했습니다 (Table 1). FID는 때때로 풀 정밀도 모델보다 더 나은 결과를 보여주었으나, 이는 낮은 양자화 노이즈 시 FID의 신뢰성 문제로 지적됩니다.
*   **지역 노이즈 보정**: 단순 min-max 양자화는 높은 FID와 낮은 $SQNR_{\theta}$로 매우 노이즈가 많은 이미지를 생성했지만, 지역 노이즈 보정(SmoothQuant 및 단일 샘플링 스텝 캘리브레이션)을 적용했을 때 이미지 품질이 크게 회복되었습니다 (Table 2). LDM 1.5 8/8 설정에서 FID가 213.72에서 24.51로, $SQNR_{\theta}$가 11.04dB에서 17.78dB로 개선되었습니다.
*   **세부 분석 (Ablations)**:
    *   **하이브리드 구성**: 출력에 가까운 업샘플링 블록이 가장 민감하며, 이 블록들에 대한 하이브리드 양자화가 가장 좋은 성능을 보였습니다 (Table 3).
    *   **지역 보정 범위**: 가장 민감한 모듈의 10%에 SmoothQuant를 적용했을 때 $SQNR_{\theta}$가 크게 증가하며, 90%까지 포화 상태를 보였습니다 (Figure 11). 불필요하게 덜 민감한 모듈에 적용하는 것은 오히려 성능 저하를 초래했습니다.
    *   **캘리브레이션 스텝 수**: 단일 샘플링 스텝 캘리브레이션은 기존의 다중 스텝 캘리브레이션보다 50배 효율적이면서도 더 높은 $SQNR_{\theta}$를 달성했습니다 (Table 4).
    *   **SQNR 계산 효율성**: 단일 순방향 패스와 적은 샘플(예: 64개)만으로도 1024개 샘플과 비교하여 0.07% 미만의 작은 차이로 SQNR을 정확하게 계산할 수 있음을 입증했습니다 (Figure 13).

## 🧠 Insights & Discussion
*   **비균일 양자화의 중요성**: LDM의 복잡한 구조와 동적인 특성으로 인해 전체 모델에 균일한 양자화를 적용하는 것은 비효율적입니다. 모델의 민감한 부분을 식별하고 맞춤형 전략을 적용하는 것이 핵심입니다.
*   **SQNR의 유효성**: SQNR은 LDM 양자화에서 민감한 블록과 모듈을 효율적으로 식별할 수 있는 강력한 지표임을 입증했습니다. 이는 FID와 같은 기존 지표의 계산 비용과 한계를 보완합니다.
*   **하이브리드 및 지역 보정의 시너지**: 전역적인 하이브리드 양자화는 누적되는 양자화 노이즈를 효과적으로 걸러내고, 지역적인 SmoothQuant는 활성화 아웃라이어로 인한 문제를 해결하여 전반적인 이미지 품질을 크게 향상시킵니다.
*   **캘리브레이션 효율성**: 단일 샘플링 스텝 캘리브레이션에 대한 통찰은 LDM 양자화의 실용성을 높이는 중요한 발견입니다. 활성화 범위의 동적인 변화를 줄여 더 안정적이고 효율적인 양자화를 가능하게 합니다.
*   **제한 사항**: FID가 낮은 양자화 노이즈에서는 이미지 품질을 정확히 반영하지 못할 수 있다는 점을 지적합니다. 또한, SmoothQuant가 이미 고정밀 양자화가 적용된 민감한 모듈에는 추가적인 큰 이점을 제공하지 않을 수 있습니다.
*   **의미**: 이 연구는 대규모 LDM을 엣지 디바이스에 배포하기 위한 실용적이고 효율적인 PTQ 전략을 제시하며, 재훈련이나 복잡한 최적화 없이도 상당한 성능 향상을 달성할 수 있음을 보여줍니다.

## 📌 TL;DR
대규모 LDM을 엣지 디바이스에 배포하기 위한 효율적인 PTQ(Post-Training Quantization) 전략을 제안합니다. 이 전략은 SQNR(Signal-to-Quantization-Noise Ratio)을 핵심 지표로 사용하여 양자화 민감도를 분석합니다. 전역적으로는 출력에 가까운 민감한 업샘플링 블록에 더 높은 정밀도 양자화를 적용하고, 지역적으로는 공간 샘플링, 트랜스포머 출력 투영, 숏컷 연결과 같은 민감 모듈의 활성화 아웃라이어를 SmoothQuant로 완화합니다. 또한, 확산 노이즈가 최고조에 달하는 단일 샘플링 스텝에서만 캘리브레이션을 수행하여 효율성을 극대화합니다. 이 통합 전략은 모델 크기와 계산량을 크게 줄이면서도 이미지 품질을 효과적으로 보존함을 입증했습니다.