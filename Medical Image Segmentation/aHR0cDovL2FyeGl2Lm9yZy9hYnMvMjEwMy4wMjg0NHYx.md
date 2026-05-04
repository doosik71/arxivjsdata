# Learning With Context Feedback Loop for Robust Medical Image Segmentation

Kibrom Berihu Girum, Gilles Créhange, and Alain Lalande (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 의료 영상 분할(Medical Image Segmentation)에서 딥러닝 모델이 생성하는 결과물이 해부학적으로 비현실적이거나 불완전하다는 점이다. 기존의 합성곱 신경망(CNN) 기반 방식은 픽셀 단위의 목적 함수(pixel-wise objective function)를 사용하기 때문에, 출력 픽셀 간의 상호 의존성(interdependence)을 충분히 학습하지 못하는 경향이 있다.

이러한 문제의 중요성은 의료 영상의 특성에서 기인한다. 의료 영상은 장기나 종양의 모양이 환자마다 매우 다양하고, 영상 모달리티에 따라 대비(contrast)가 낮거나 금속 아티팩트(artifact)가 발생하는 등 노이즈가 많다. 특히, 기존 CNN은 형태(shape)보다는 질감(texture)을 인식하는 편향(bias)을 가지고 있어, 결과적으로 분할 영역 내부에 구멍이 생기거나 해부학적으로 불가능한 형태의 결과물이 도출되는 문제가 발생한다. 따라서 본 논문의 목표는 이러한 컨텍스트 정보와 지역 간 관계를 효과적으로 학습하여, 저대비 영상에서도 강건(robust)하고 해부학적으로 타당한(anatomically plausible) 분할 결과를 생성하는 새로운 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 분할 문제를 두 개의 시스템이 상호작용하는 **재귀적 프레임워크(recurrent framework)**로 공식화하는 것이다. 이를 **LFB-Net (Learning with context FeedBack system)**이라고 명명하며, 주요 설계 직관은 다음과 같다.

1.  **피드백 루프(Feedback Loop)의 도입**: 단순히 입력 영상을 출력 영상으로 매핑하는 feed-forward 방식에서 벗어나, 한 번 예측된 결과물을 다시 네트워크의 입력으로 활용하여 오류를 수정하는 루프를 설계하였다.
2.  **두 시스템의 분리 및 통합**:
    *   **Forward System**: 원본 영상에서 초기 분할 맵을 예측하는 주 모듈이다.
    *   **Feedback System**: 예측된 확률 맵(probabilistic output)을 입력받아 고차원 특징 공간(high-level feature space)으로 변환하여 Forward System에 다시 제공하는 정규화(regularizer) 역할을 수행한다.
3.  **암시적 형태 지식 학습**: 명시적인 shape prior를 설계하여 주입하는 대신, 피드백 루프를 통해 네트워크가 스스로 픽셀 간의 관계와 형태적 특징을 학습하도록 하여 모델의 범용성을 높였다.

## 📎 Related Works

논문에서는 기존의 의료 영상 분할 접근 방식을 다음과 같이 분류하고 그 한계를 지적한다.

-   **전통적 방식**: Edge detection, Level-set, Active shape model 등이 사용되었으나, 사람이 직접 설계한 특징(hand-crafted features)에 의존하므로 새로운 케이스에 적응하기 어렵다.
-   **CNN 기반 방식**: U-Net, FCN, ResU-Net 등이 등장하여 계층적 특징 추출을 자동화하였다. 특히 U-Net의 skip-connection은 공간 정보 손실을 줄였으나, 여전히 픽셀 단위 손실 함수로 인해 출력 픽셀 간의 상관관계를 놓치는 한계가 있다.
-   **형태 제약 방식(Shape Priors)**: 통계적 모양 모델이나 multi-task learning을 통해 형태 정보를 강제하는 연구들이 진행되었다. 하지만 이러한 방법들은 타겟에 대한 명시적인 사전 지식이 필요하며, 모델링 과정이 복잡하다.
-   **후처리 방식(Post-processing)**: Denoising auto-encoder 등을 통해 결과물을 정제하는 방식이 제안되었으나, 후처리 단계에서는 원본 영상을 참조하지 않으므로 초기 분할 결과가 너무 잘못되었을 경우 이를 바로잡지 못하는 한계가 있다.

LFB-Net은 이러한 한계를 극복하기 위해 피드백 루프를 통해 원본 영상의 특징과 예측 결과의 컨텍스트를 동시에 고려하는 방식을 취한다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
LFB-Net은 **Forward System ($S$)**과 **Feedback System ($F$)**이라는 두 개의 상호 연결된 네트워크로 구성된다.

#### 1. Forward System ($S$)
수정된 U-Net 아키텍처를 사용하며, 입력 영상 $x$를 받아 예측 라벨 $\hat{y}$를 생성한다.
$$\hat{y} = S(x) = S_d(S_e(x)) \quad (1)$$
여기서 $S_e$는 인코더(encoder), $S_d$는 디코더(decoder)이다. 인코더는 영상을 잠재 공간(latent space) $h_s \in \mathbb{R}^{W/d \times H/d \times C}$로 매핑한다.

#### 2. Feedback System ($F$)
FCN(Fully Convolutional Network) 기반으로, Forward System의 출력인 확률 맵 $\hat{y}$를 입력받아 다시 고차원 특징 공간 $h_f$로 변환한다.
$$\hat{\hat{y}} = F(\hat{y}) = F_d(F_e(\hat{y})) \quad (2)$$
여기서 $F_e$는 피드백 인코더이며, 이를 통해 생성된 $h_f$는 Forward System의 디코더로 다시 전달된다.

#### 3. 통합 및 재귀적 절차 (Integration)
두 시스템은 다음과 같은 재귀적 프로세스로 통합된다.
1.  $i$번째 반복에서 Forward System이 $\hat{y}_i$를 생성한다.
2.  Feedback System의 인코더가 $\hat{y}_i$를 입력받아 피드백 잠재 공간 $h_f^i$를 생성한다: $h_f^i = F_e(\hat{y}_i) \quad (3)$.
3.  $i+1$번째 반복에서 Forward System의 디코더는 원본 영상의 특징 $h_s^i$와 피드백 특징 $h_f^i$를 함께 입력받아 최종 결과를 도출한다:
$$\hat{y}_{i+1} = S_{d, i+1}(h_s^i, h_f^i) \quad (5)$$
이때 두 잠재 공간은 **Concatenation(결합)** 방식으로 병합된다.

### 학습 절차 (Training Strategy)
학습은 다음과 같은 3단계 교차 전략으로 진행되며, 수렴할 때까지 반복한다.
-   **Step 1**: 피드백 없이($h_f=0$) Forward System의 가중치 $w_s^i$를 학습시킨다.
-   **Step 2**: Forward System의 출력 $\hat{y}$를 입력으로 하여 Feedback System의 가중치 $w_f^i$를 학습시킨다.
-   **Step 3**: Forward System의 인코더와 Feedback System의 인코더를 고정(freeze)하고, 결합된 특징 $(h_s, h_f)$를 사용하는 Forward System의 **디코더 부분($w_{sd}^{i+1}$)**만 학습시킨다.

### 손실 함수 (Loss Function)
Binary Cross-Entropy ($L_1$)와 Dice Coefficient Loss ($L_2$)의 평균을 사용하여 학습한다.
$$L_{total} = \frac{1}{2} \times (L_1 + L_2) \quad (6)$$
-   $L_1$은 클래스별 픽셀 예측의 정확도를 측정하며, $L_2$는 예측 영역과 실제 영역의 중첩도를 측정하여 클래스 불균형 문제를 완화한다.

### 네트워크 세부 아키텍처
-   **Forward System**: $3 \times 3$ Convolution, ELU 활성화 함수, Batch Normalization 및 **Squeeze-and-Excitation (SE) network**를 적용하여 채널 간 특징을 보정한다.
-   **Feedback System**: Learning deconvolution network 기반의 FCN 구조를 사용한다.

## 📊 Results

### 실험 설정
-   **데이터셋**: 전립선 CT(78케이스), 내이 $\mu$CT(Hear-EU), 심장 Cine-MRI(ACDC), 심장 초음파(CAMUS) 등 4가지의 서로 다른 임상 데이터셋을 사용하였다.
-   **비교 대상**: U-Net, ResU-Net, FCN, Post-DAE, Attention Gated Network (AGN) 등.
-   **측정 지표**: Dice Similarity Coefficient (DSC), Hausdorff Distance (HD), Relative Volume Difference (RVD).

### 주요 결과
-   **정량적 성과**: 모든 데이터셋에서 LFB-Net이 SOTA 방법들보다 우수한 성능을 보였다. 특히 전립선 CT에서 U-Net 대비 HD 값을 11mm 감소시켰으며, 내이 $\mu$CT에서는 DSC를 4% 향상시켰다.
-   **심장 구조 분할**: ACDC 및 CAMUS 데이터셋에서 특히 분할이 어려운 우심실(RV)과 심근(MYO), 좌심방(LA) 영역에서 유의미한 성능 향상($p < 0.05$)을 보였다.
-   **정성적 분석**: 기존 모델들이 결과물에 구멍(hole)을 만들거나 비현실적인 형태를 생성하는 반면, LFB-Net은 해부학적으로 타당하고 매끄러운 분할 결과를 생성하였다.
-   **효율성**: U-Net(32M 파라미터)보다 적은 파라미터 수(학습 시 8.5M, 테스트 시 7.9M)를 가지며, 추론 속도가 매우 빨라(Cine-MRI 기준 0.025s) 실시간 적용 가능성을 입증하였다.

### Ablation Study 결과
-   **피드백 루프의 효과**: 피드백 루프가 없는 Forward System(FS)보다 DSC와 HD 모든 지표에서 성능이 향상되었으며, 특히 어려운 케이스에서 오차의 최대값을 크게 줄였다.
-   **병합 전략**: 잠재 공간 $h_s$와 $h_f$를 합치는 방법 중 **Concatenation**이 Addition이나 Multiplication보다 HD 지표에서 유의미하게 우수하였다.
-   **학습 곡선**: 피드백 루프를 사용했을 때 학습 및 검증 손실(loss)이 더 빠르게 감소하며, Overfitting 경향이 줄어드는 것이 확인되었다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 **'피드백 루프'라는 제어 시스템 이론의 개념을 의료 영상 분할에 성공적으로 도입**했다는 점이다. 기존의 shape prior 방식이 전문가의 사전 지식을 수동으로 주입해야 했던 반면, LFB-Net은 예측 결과를 다시 입력으로 사용하는 구조를 통해 네트워크가 암시적으로 형태적 특성을 학습하게 만들었다.

특히 주목할 점은 **어려운 구조(complex structures)에 대한 강건함**이다. 심장의 우심실이나 심근처럼 경계가 모호하고 형태 변화가 심한 영역에서 성능 향상이 두드러진 것은, 피드백 루프가 일종의 '교정 단계' 역할을 하여 초기 예측의 불확실성을 효과적으로 제거했음을 시사한다. 또한, 파라미터 수를 줄이면서도 성능을 높인 점은 의료 현장의 제한된 컴퓨팅 자원 환경에서 큰 이점이 된다.

다만, 본 논문에서 제시된 피드백 루프의 횟수가 고정적이라는 점과, 2D 기반의 접근 방식을 3D 데이터에 적용할 때의 토폴로지 캡처 능력에 대해서는 향후 연구 과제로 남겨두고 있다.

## 📌 TL;DR

LFB-Net은 **Forward System(U-Net)**과 **Feedback System(FCN)**을 결합하여, 예측된 분할 결과를 다시 특징 공간으로 변환해 피드백하는 재귀적 프레임워크이다. 이를 통해 명시적인 shape prior 없이도 **해부학적으로 타당하고 강건한 분할 결과**를 얻었으며, 특히 저대비 영상이나 복잡한 장기 구조 분할에서 기존 SOTA 모델들을 능가하는 성능과 효율성을 입증하였다. 이 연구는 향후 실시간 의료 영상 분석 및 복잡한 3D 해부학 구조 분석에 중요한 기초가 될 것으로 보인다.