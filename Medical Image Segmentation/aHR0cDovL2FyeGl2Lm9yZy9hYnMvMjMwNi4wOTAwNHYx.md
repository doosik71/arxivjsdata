# Annotator Consensus Prediction for Medical Image Segmentation with Diffusion Models

Tomer Amit, Shmuel Shichrur, Tal Shaharbany, and Lior Wolf (2023)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 의료 영상 분할(Medical Image Segmentation) 작업에서 발생하는 전문가 간(inter-observer) 및 전문가 내(intra-observer)의 주석 변동성(variability) 문제이다. 동일한 영상에 대해 여러 명의 전문가가 서로 다른 분할 결과를 제공하는 경우가 많으며, 이는 모델 학습을 위한 정답지(ground truth)를 설정하는 데 큰 어려움을 준다.

이러한 변동성은 전문가의 경험, 전문성, 그리고 주관적인 판단 차이에서 기인한다. 또한, 수동으로 주석을 생성하는 과정은 막대한 시간과 비용이 소요되므로 확장성이 떨어진다는 한계가 있다. 따라서 본 연구의 목표는 여러 전문가의 주석 정보를 효과적으로 융합하여, 전문가들의 합의(consensus)를 반영한 정확하고 일관된 통합 분할 맵(unified segmentation map)을 생성하는 multi-expert prediction 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 Diffusion Model을 활용하여 각 합의 수준(consensus level)에 해당하는 이진 맵을 생성하고, 이를 평균 내어 최종적인 soft map을 도출하는 것이다.

핵심적인 설계 직관은 합의 수준을 조정하는 것이 '매우 보수적인 분할 맵'에서 '매우 자유로운 분할 맵'으로의 동적인 변화를 만드는 것과 같다는 점이며, Diffusion Model이 이러한 조건부 생성을 매우 잘 수행할 수 있다는 점을 이용하였다. 또한, 추론 과정에서 여러 번의 생성 과정을 거쳐 결과를 평균 내는 방식을 통해 결과의 안정성을 높이고 성능을 향상시켰다.

## 📎 Related Works

### 기존 접근 방식 및 한계
- **다수결 투표(Majority Voting), 레이블 융합(Label Fusion), 레이블 샘플링(Label Sampling):** 여러 주석을 통합하려는 시도가 있었으나, 전문가 간의 복잡한 변동성을 정교하게 캡처하는 데 한계가 있다.
- **확률적 모델(Probabilistic Models):** U-Net과 Conditional VAE를 결합하거나 Diffusion Model과 KL divergence를 결합하여 변동성을 캡처하려는 시도가 있었으나, 본 논문은 합의 맵(consensus maps) 자체를 ground truth로 사용하는 차별점을 가진다.

### Diffusion Probabilistic Models (DPM) 및 차별점
- Diffusion 모델은 가우시안 분포와 같은 단순한 분포를 복잡한 데이터 분포로 변환하는 생성 모델이다.
- 기존의 Diffusion 기반 분할 방법(예: Wolleb et al.)과 비교하여, 본 논문은 (i) 조건 신호(condition signal)의 결합 방법과 (ii) 조건 신호를 처리하는 인코더 구조에서 차이가 있으며, 더 적은 diffusion step($T$)을 사용하여 실행 시간을 단축하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 합의 맵 정의
본 방법론은 입력 영상 $I \in \mathbb{R}^{W \times H}$, 시간 단계 $t$, 그리고 합의 지표 $c$를 조건으로 하는 Diffusion Model을 사용한다. $C$명의 전문가가 제공한 주석 $\{A_k^i\}_{i=1}^C$가 있을 때, 합의 수준 $c$에서의 ground truth 합의 맵 $M_k^c$는 다음과 같이 정의된다.

$$M_k^c[x,y] = \begin{cases} 1 & \text{if } \sum_{i=1}^C A_k^i[x,y] \geq c \\ 0 & \text{otherwise} \end{cases}$$

즉, 특정 픽셀을 정답으로 표시한 전문가의 수가 $c$명 이상일 때만 해당 픽셀을 1로 설정하는 이진 맵이다.

### 학습 절차 및 손실 함수
학습 단계에서는 합의 수준 $c \sim U[1, 2, \dots, C]$와 시간 단계 $t \sim U[1, \dots, T]$를 무작위로 샘플링한다. 정규 분포에서 샘플링된 노이즈 $X^T$를 사용하여 $t$ 단계의 노이즈 맵 $x_t$를 생성한다.

$$x_t = \sqrt{\bar{\alpha}_t} M_k^c + \sqrt{1 - \bar{\alpha}_t} X^T, \quad X^T \sim \mathcal{N}(0, \mathbf{I}_{n \times n})$$

모델 $\epsilon_\theta$는 이 $x_t$와 조건들($I_k, z_t, z_c$)을 입력받아 추가된 노이즈 $\epsilon$을 예측하며, 손실 함수는 다음과 같은 MSE(Mean Squared Error)를 최소화하는 방향으로 학습된다.

$$\mathbb{E}_{x_0, \epsilon, x_e, t, c} [\|\epsilon - \epsilon_\theta(x_t, I_k, z_t, z_c)\|^2]$$

### 네트워크 아키텍처
전체 구조는 U-Net을 기반으로 하며, 인코더 부분이 세 개의 네트워크 $E, F, G$로 세분화되어 있다.
- **$G$ (Image Encoder):** 입력 영상을 처리하며, Batch Normalization이 없는 RRDB(Residual in Residual Dense Blocks)를 사용하여 특징을 추출한다.
- **$F$ (Map Encoder):** 현재 단계의 세그멘테이션 맵 $x_t$를 인코딩한다.
- **$E, D$ (U-Net Encoder/Decoder):** $F$와 $G$에서 나온 신호의 합 $u_t = F(x_t, z_c, z_t) + G(I_k)$를 입력받아 처리한다. 각 레벨은 Residual Block과 Attention Layer로 구성되어 있으며, Skip Connection이 적용되어 있다.

### 추론 및 앙상블 절차
추론 시에는 모든 가능한 합의 수준 $c = 1, \dots, C$에 대해 각각 모델을 실행하여 $x_0$를 얻은 후, 이를 평균 내어 최종 soft-label map을 생성한다.

$$\text{Final Prediction} = \frac{\sum_{i=1}^C x_0^i}{C}$$

또한, Diffusion 모델의 확률적 특성으로 인해 발생하는 변동성을 줄이기 위해, 동일한 입력에 대해 추론을 25번 반복 수행하고 그 결과들을 평균 내는 방식을 사용한다.

## 📊 Results

### 실험 설정
- **데이터셋:** QUBIQ 벤치마크 (Kidney, Brain, Tumor, Prostate 1, Prostate 2 등 5개 데이터셋)
- **평가 지표:** Soft Dice coefficient (임계값 0.1, 0.3, 0.5, 0.7, 0.9에서 측정)
- **비교 대상:** FCN, MCD, FPM, DAF, MV-UNet, LS-UNet, MH-UNet, MRNet, AMIS, DMISE 등

### 주요 결과
- **정량적 성과:** 제안 방법은 모든 QUBIQ 데이터셋에서 기존 SOTA 방법들보다 높은 Soft Dice 점수를 기록하였다. 특히 Kidney 데이터셋에서 96.58로 매우 높은 성능을 보였다.
- **Ablation Study:** 
    - **Consensus $\gg$ Annotator $>$ No annotator $\gg$ Soft-label:** 각 합의 수준을 개별적으로 학습하고 평균 내는 'Consensus' 방식이, 전문가별 맵을 예측하거나 직접적으로 soft-label(비율 맵)을 예측하는 방식보다 월등히 성능이 좋았다.
    - **생성 횟수 영향:** 생성된 이미지의 수를 늘릴수록 성능이 향상되는 경향을 보였으며, 데이터셋에 따라 최적의 반복 횟수(5~25회)가 달랐다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구는 Diffusion Model이 조건부 생성 능력을 통해 합의 수준의 변화(보수적 $\leftrightarrow$ 자유적)를 매우 정교하게 캡처할 수 있음을 입증하였다. 특히, 전문가 간의 합의 정도(Pairwise Dice score)가 높은 데이터셋(Kidney, Prostate 1)에서 모델의 성능이 더욱 향상되는 경향을 보였다. 이는 모델이 전문가들의 공통된 의견을 효과적으로 학습하고 있음을 시사한다.

### 한계 및 논의
- **추론 시간:** Diffusion 모델의 특성상 여러 단계($T=100$)를 거쳐야 하고, 특히 본 논문에서는 각 합의 수준별 생성과 25회의 반복 생성을 수행하므로 추론 시간이 상당히 길어질 수 있다.
- **전문가 수와의 상관관계:** 흥미롭게도 전문가의 수와 모델 성능 사이에는 유의미한 상관관계가 발견되지 않았다. 이는 전문가의 수보다 전문가들 사이의 '합의 수준' 자체가 더 중요한 요인임을 의미한다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 전문가들 사이의 주석 불일치 문제를 해결하기 위해, **합의 수준(Consensus Level)을 조건으로 하는 Diffusion Model**을 제안하였다. 각 합의 단계별 이진 맵을 생성한 뒤 이를 평균 내어 전문가들의 합의가 반영된 soft-label map을 도출하는 방식이며, 이를 통해 QUBIQ 벤치마크의 모든 데이터셋에서 기존 SOTA 모델들을 뛰어넘는 성능을 달성하였다. 이 연구는 다수 전문가의 의견을 통합해야 하는 의료 AI 시스템의 신뢰성을 높이는 데 중요한 기여를 할 수 있다.