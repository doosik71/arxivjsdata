# Wav2vec-C: A Self-supervised Model for Speech Representation Learning
Samik Sadhu, Di He, Che-Wei Huang, Sri Harish Mallidi, Minhua Wu, Ariya Rastrow, Andreas Stolcke, Jasha Droppo, Roland Maas

## 🧩 Problem to Solve
기존의 자기 지도 학습 모델인 wav2vec 2.0은 음성 인식을 위한 이산적인 음성 표현을 학습하지만, 벡터 양자화 (Vector Quantization, VQ) 모듈의 코드북 학습 과정에서 몇 가지 문제가 발생할 수 있습니다. 특히, 음성 활동 감지(VAD) 코드북(음성/비음성 이진 할당)이나 시간적으로 불변하는 코드북과 같이 "국소 최적화된 코드북(locally optimal codebooks)"이 형성되어 코드북 활용률이 낮아지고, 결과적으로 모델이 다양한 음성 단위를 효과적으로 학습하지 못하는 문제가 있습니다. 이 연구는 이러한 코드북 활용 문제를 해결하고 더 강력하며 일반화 가능한 음성 표현을 학습하는 것을 목표로 합니다.

## ✨ Key Contributions
*   **Wav2vec-C 모델 제안**: wav2vec 2.0과 VQ-VAE의 아이디어를 결합하여 코드북 활용 문제를 해결하는 새로운 자기 지도 학습 모델인 wav2vec-C를 제안합니다.
*   **실제 원거리 음성 데이터 활용**: 문헌의 대부분 연구가 깨끗한 읽기 음성(clean read speech)을 사용하는 것과 달리, -40dB에서 50dB에 이르는 다양한 SNR(신호 대 잡음비)을 가진 실제 원거리 음성 명령 및 질의 데이터를 사용하여 모델을 훈련하고 평가했습니다.
*   **대량의 레이블 데이터에 대한 자기 지도 학습 적용**: 자기 지도 학습의 효과는 레이블 데이터의 양이 증가할수록 감소하는 경향이 있다는 기존의 관찰과 달리, 1천 시간의 상대적으로 많은 양의 레이블 데이터를 사용하여 자기 지도 학습의 적용 가능성을 탐색했습니다.
*   **생산 수준의 ASR 모델을 위한 소형 모델**: 문헌에서 제안되는 자기 지도 학습 모델의 대형화 추세와는 반대로, 저지연(low-latency) 생산 수준의 ASR 모델 구현을 위해 모델 크기를 제한했습니다.
*   **다양한 VQ 프레임워크 비교**: Gumbel-softmax와 K-means와 같은 다른 벡터 양자화 프레임워크가 강건성(robustness)과 코드북 활용률에 미치는 영향을 탐색하고 비교했습니다.

## 📎 Related Works
이 연구는 다음을 포함한 여러 기존 연구에서 영감을 받거나 이를 기반으로 합니다.
*   **wav2vec 2.0**: 마스킹된 이산 음성 인코딩을 예측하기 위해 컨텍스트 표현을 사용하는 자기 지도 학습 모델입니다. Wav2vec-C의 주요 기반이 됩니다.
*   **VQ-VAE (Vector Quantized-Variational AutoEncoder)**: 이산적인 표현 학습을 위한 모델로, wav2vec-C의 일관성 네트워크(consistency network) 아이디어에 영향을 주었습니다.
*   **SpecAugment**: 음성 입력 특징에 대한 데이터 증강 기법으로, 이 연구에서도 마스킹(masking)을 위해 사용됩니다.
*   **RNN-T (Recurrent Neural Network Transducer)**: 배포 가능한 종단 간 음성 인식 시스템에 널리 사용되는 모델로, 이 연구에서는 자기 지도 학습된 모델의 성능 평가를 위한 백본(backbone) ASR 모델로 사용됩니다.
*   **다른 자기 지도 학습 모델**: wav2vec, CPC (Contrastive Predictive Coding), ALBERT 등 다양한 자기 지도 학습 패러다임이 언급됩니다.

## 🛠️ Methodology
Wav2vec-C는 wav2vec 2.0 아키텍처를 기반으로 하며, 코드북 학습을 강화하기 위해 추가적인 일관성 네트워크를 도입합니다.

1.  **입력 특징**: 모델은 로그 단기 푸리에 변환(log-STFT) 특징 $X = [x_1, x_2, ..., x_T]$를 입력으로 받습니다.

2.  **인코더 네트워크 ($f$)**:
    *   입력 특징 $X$를 잠재 임베딩 $Z = [z_1, z_2, ..., z_T]$로 매핑합니다.
    *   3계층 LSTM (hidden dimension 768)으로 구성됩니다.
    *   훈련 중 코드북 안정화를 돕기 위해 인코더 기울기는 $\gamma=0.1$로 스케일링됩니다.

3.  **마스킹 (SpecAugment)**:
    *   인코딩된 연속 임베딩 $Z$의 일부를 무작위로 마스킹하여 $Z_{\text{masked}}$를 생성합니다.
    *   각 발화에 5개의 마스크를 사용하며, 발화 길이의 최대 16% 너비를 가집니다. 평균적으로 인코딩된 프레임의 40%가 마스킹됩니다.

4.  **벡터 양자화 (VQ) 모듈 ($q$)**:
    *   인코딩된 $Z$를 양자화된 표현 $\hat{Z} = [\hat{z}_1, \hat{z}_2, ..., \hat{z}_T]$로 변환합니다.
    *   **제품 양자화(Product Quantization)**를 사용하며, $G=2$개의 코드북 $Q = [Q^{(1)}, Q^{(2)}]$을 가집니다. 각 코드북 $Q^{(i)} \in \mathbb{R}^{V \times K}$는 $V=320$개의 코드와 $K=384$차원을 가집니다.
    *   **두 가지 VQ 기법을 탐색**:
        *   **Gumbel-softmax**: 각 분할된 임베딩 $z^{(i)}$를 선형 변환하여 로짓(logits)을 생성하고, 이를 Gumbel-softmax를 통해 $V$개의 코드에 대한 하드 분포로 변환하여 미분 가능한 코드 선택을 가능하게 합니다. 코드북 붕괴를 방지하기 위해 다양성 손실 $L_d$를 사용합니다.
        *   **K-means**: 순방향 전달 시 $z^{(i)}$에 가장 가까운 코드 $e$를 선택합니다 ($\hat{z}^{(i)} = \text{arg min}_{e \in Q^{(i)}} ||z^{(i)} - e||^2$). 역방향 전달 시에는 straight-through estimator를 사용하여 기울기를 직접 연속 임베딩 $z$로 복사합니다. 커밋먼트 손실(commitment loss) $L_k$를 포함합니다.

5.  **컨텍스트 네트워크 ($g$)**:
    *   마스킹된 임베딩 $Z_{\text{masked}}$를 처리하여 컨텍스트 표현 $C = [c_1, c_2, ..., c_T]$를 생성합니다.
    *   5개의 트랜스포머 계층(모델 차원 1024, 피드포워드 차원 4096, 16개 어텐션 헤드)으로 구성됩니다.

6.  **대조 손실 ($L_m$)**:
    *   컨텍스트 표현 $C$와 양자화된 임베딩 $\hat{Z}$ 사이의 대조 점수를 최대화합니다.
    *   $$ L_m = -\log \frac{\exp(d(c_t, \hat{z}_t))/\kappa}{\sum_{z \in \Theta} \exp(d(c_t, z))/\kappa} $$
        여기서 $d$는 코사인 유사도, $\Theta$는 $\hat{z}_t$와 $N=50$개의 음성 샘플, $\kappa$는 온도 변수입니다.

7.  **일관성 네트워크 ($r$)**:
    *   양자화된 임베딩 $\hat{Z}$를 일관성 벡터 $S = [s_1, s_2, ..., s_T]$로 매핑합니다.
    *   3계층 LSTM으로 구성됩니다.

8.  **일관성 손실 ($L_c$)**:
    *   일관성 벡터 $S$와 원래 입력 특징 $X$ 사이의 $L_2$ 노름 거리(normed distance)를 최소화합니다.
    *   $$ L_c = ||x_t - s_t||^2 $$

9.  **총 손실 ($L$)**:
    *   훈련 시 주 대조 손실($L_m$), 코드북 손실($L_{cb}$), 일관성 손실($L_c$)을 함께 최소화합니다.
    *   $$ L = L_m + L_{cb} + \gamma L_c $$
        여기서 $\gamma$는 일관성 손실의 가중치로, $\gamma=0$일 경우 wav2vec 2.0 모델이 되고, $\gamma=1$일 경우 wav2vec-C 모델이 됩니다. $L_{cb}$는 VQ 유형에 따라 달라집니다 (Gumbel-softmax는 다양성 손실, K-means는 K-means 손실).

10. **파인튜닝**: 자기 지도 학습 모델 훈련 후, 컨텍스트 네트워크 $g$의 출력을 RNN-T ASR 모델의 음성 인코더로 사용하여 1천 시간의 레이블 데이터로 파인튜닝합니다.

## 📊 Results
*   **rWERR (relative Word Error Rate Reduction) 성능**:
    *   Wav2vec-C (GS) 모델은 깨끗한 테스트 세트(SNR$_{20}$, SNR$_{16}$)에서 각각 1.6% 및 1.2%의 rWERR을 달성하며 긍정적인 성능 향상을 보였습니다.
    *   전반적으로 wav2vec-C는 평균 1.4%의 rWERR을 달성하여, wav2vec 2.0의 0.7% rWERR에 비해 두 배의 오류 감소를 보였습니다.
    *   wav2vec 2.0 모델은 깨끗한 테스트 세트에서 성능 향상이 미미했으나, 노이즈가 있는 테스트 세트에서는 일부 이득을 보였습니다 (wav2vec 2.0 (KM)이 wav2vec 2.0 (GS)보다 우수).
    *   wav2vec-C (KM)은 wav2vec-C (GS)에 비해 노이즈가 있는 테스트 세트에서 더 나은 강건성을 보였지만, 깨끗한 테스트 세트에서는 wav2vec-C (GS)가 더 우수했습니다.

*   **코드북 활용률**:
    *   wav2vec-C (GS)는 코드북을 100% 활용하는 압도적인 결과를 보였습니다. 이는 일관성 손실과 다양성 손실의 결합이 모델이 다양한 코드를 선택하도록 강제하기 때문입니다.
    *   반면, wav2vec 2.0 (GS)는 15%의 활용률을 보였고, K-means 기반 모델(wav2vec 2.0 (KM), wav2vec-C (KM))은 1% 미만의 매우 낮은 활용률을 보였습니다.
    *   낮은 코드북 활용률에도 불구하고 K-means VQ는 특히 노이즈가 있는 환경에서 더 강건한 성능을 보였습니다.

## 🧠 Insights & Discussion
*   **일관성 손실의 중요성**: Wav2vec-C의 핵심인 일관성 손실은 양자화된 표현이 입력 특징에 대한 의미 있는 정보를 보존하도록 강제하며, 이는 코드북 활용률을 크게 향상시키고 전반적인 ASR 성능을 개선하는 데 기여합니다. 특히 Gumbel-softmax VQ와 결합될 때 최상의 코드북 활용률을 달성했습니다.
*   **강건성과 코드북 다양성**: ASR의 강건성은 코드북 다양성과 관련이 있음을 보여줍니다. wav2vec-C (GS)는 높은 코드북 활용률로 깨끗한 음성에서 뛰어난 성능을 보였지만, 노이즈가 있는 환경에서는 강건성 측면에서 다소 손실이 있었습니다. 반면, K-means VQ는 코드북 활용률이 낮음에도 불구하고 노이즈가 있는 데이터에 대해 더 강건한 특성을 보였습니다. 이는 특정 응용 도메인(예: 높은 강건성 요구)에서는 코드북 다양성이 반드시 높은 것이 최적의 설계 선택이 아닐 수 있음을 시사합니다.
*   **실제 데이터 및 대규모 레이블 데이터에 대한 효과**: 실제 원거리 노이즈 데이터와 비교적 대량의 레이블 데이터(1천 시간)를 사용한 훈련에서도 자기 지도 학습의 유효성을 입증했습니다. 이는 이전 연구들에서 제기되었던 레이블 데이터 양 증가에 따른 자기 지도 학습 효과 감소 경향을 극복할 수 있음을 보여줍니다.
*   **생산 환경 적용 가능성**: 대규모 모델이 아닌, 생산 수준의 ASR 모델에 적합한 작은 모델 크기로도 상당한 성능 향상을 달성하여, 실제 서비스에 적용될 가능성을 제시합니다.

## 📌 TL;DR
wav2vec-C는 wav2vec 2.0의 코드북 활용 문제를 해결하기 위해 VQ-VAE의 일관성 개념을 도입한 새로운 자기 지도 음성 표현 학습 모델입니다. 이 모델은 양자화된 표현으로부터 원본 입력 특징을 재구성하도록 학습하는 "일관성 네트워크"를 추가하여, 코드북이 더 풍부하고 의미 있는 정보를 포착하도록 강제합니다. 실제 원거리 노이즈 데이터와 1천 시간의 레이블 데이터를 사용한 RNN-T ASR 모델 파인튜닝 결과, wav2vec-C는 기준 모델 대비 평균 1.4%의 WER 감소를 달성하여 wav2vec 2.0 (0.7% 감소)보다 뛰어난 성능을 보였습니다. 특히, Gumbel-softmax VQ와 결합된 wav2vec-C는 100%의 코드북 활용률을 달성하여 효율적인 코드 학습을 입증했으며, 코드북 다양성과 ASR 강건성 간의 상관관계를 강조합니다.