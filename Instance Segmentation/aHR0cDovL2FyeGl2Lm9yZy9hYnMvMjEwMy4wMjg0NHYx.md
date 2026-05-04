# Learning With Context Feedback Loop for Robust Medical Image Segmentation

Kibrom Berihu Girum, Gilles Cr ehange, and Alain Lalande (2021)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation)에서 딥러닝 모델이 가지는 근본적인 한계점을 해결하고자 한다. 기존의 Convolutional Neural Networks (CNN) 기반 접근 방식은 주로 픽셀 단위의 목적 함수(pixel-wise objective function)를 사용하여 학습한다. 이러한 방식은 다음과 같은 문제를 야기한다.

- **출력 픽셀 간의 상호 의존성 부족**: 픽셀 단위로만 학습하기 때문에 분할 결과가 불완전하거나 해부학적으로 비현실적인(unrealistic) 결과가 도출될 가능성이 크다.
- **질감 편향(Texture Bias)**: CNN은 형태(Shape)보다는 질감(Texture)을 인식하는 경향이 강하여, 의료 영상과 같이 대비가 낮거나 노이즈가 많은 환경에서 견고한 성능을 내기 어렵다.
- **해부학적 정보의 부재**: 기존의 Shape Prior를 통합하려는 시도들은 명시적인 사전 지식이 필요하거나, 입력 영상을 고려하지 않는 후처리(post-processing) 방식이라는 한계가 있다.

따라서 본 연구의 목표는 명시적인 Shape Prior 없이도 해부학적으로 타당하고(anatomically plausible) 견고한 분할 결과를 생성할 수 있는 새로운 재귀적 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 분할 문제를 두 개의 상호 연결된 시스템으로 구성된 **재귀적 프레임워크(Recurrent Framework)**로 공식화하는 것이다. 이를 **LFB-Net (Learning with context FeedBack system)**이라 명명하며, 주요 기여 사항은 다음과 같다.

- **Context Feedback Loop 도입**: 예측된 분할 결과(probabilistic output)를 다시 네트워크의 입력으로 피드백하여, 모델이 자신의 이전 실수를 수정하고 고수준의 문맥 정보를 학습할 수 있도록 하는 루프 구조를 설계하였다.
- **두 시스템의 협력 구조**: raw 이미지를 처리하는 'Forward System'과 예측 결과를 분석하여 고차원 특징 공간으로 변환하는 'Feedback System'을 분리하여 설계함으로써 학습의 안정성과 정확도를 높였다.
- **해부학적 타당성 확보**: 명시적인 Shape Prior나 별도의 후처리 과정 없이도, 피드백 루프를 통해 출력 픽셀 간의 상호 의존성을 학습함으로써 해부학적으로 타당한 결과를 얻어냈다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들의 흐름과 한계를 설명한다.

- **전통적 방식 및 초기 CNN**: Edge detection, Level-set, Shape models 등은 수동으로 설계된 특징(hand-crafted features)에 의존하여 일반화 능력이 떨어진다. 이후 U-Net과 같은 Encoder-Decoder 구조가 등장하며 자동 특징 추출이 가능해졌다.
- **Shape Prior 통합 시도**: 다중 작업 학습(Multi-task learning)이나 통계적 형태 모델을 통해 해부학적 제약 조건을 추가하려는 시도가 있었으나, 이는 타겟에 대한 명시적인 사전 지식이 필요하다는 제약이 있다.
- **후처리 방식 (Post-processing)**: Denoising Autoencoder (DAE) 등을 이용해 결과를 다듬는 방식은 입력 영상의 원본 정보를 참조하지 않으므로, 초기 분할 결과가 너무 잘못되었을 경우 이를 회복하기 어렵다.
- **Attention 및 Recurrent 네트워크**: Gated Attention이나 RNN을 통해 문맥 정보를 강화하려는 시도가 있었으나, 여전히 Feed-forward 방식의 한계로 인해 질감 특징에 치우치는 경향이 있다.

LFB-Net은 이러한 기존 방식들과 달리, 예측된 결과 자체를 다시 고차원 특징 공간으로 인코딩하여 피드백함으로써 모델이 스스로 결과를 검토하고 수정하는 '두 번째 기회'를 제공한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
LFB-Net은 **Forward System ($S$)**과 **Feedback System ($F$)**의 두 가지 네트워크로 구성된다.

- **Forward System ($S$)**: 입력 영상 $x$를 받아 분할 맵 $\hat{y}$를 예측하는 수정된 U-Net 구조이다.
- **Feedback System ($F$)**: Forward System이 출력한 확률적 결과 $\hat{y}$를 입력받아, 이를 다시 고수준의 특징 공간 $h_f$로 변환하여 Forward System의 디코더에 전달하는 FCN(Fully Convolutional Network) 구조이다.

### 2. 상세 프로세스 및 방정식
시스템의 작동 과정은 다음과 같이 정의된다.

**Step 1: Forward System의 기본 예측**
입력 이미지 $x$에 대해 엔코더 $S_e$와 디코더 $S_d$를 거쳐 예측값 $\hat{y}$를 생성한다.
$$\hat{y} = S(x) = S_d(S_e(x))$$

**Step 2: Feedback System의 특징 추출**
Forward System의 출력 $\hat{y}$를 Feedback System $F$의 엔코더 $F_e$에 통과시켜 고차원 특징 공간 $h_f$를 얻는다.
$$h_f^i = F_e(\hat{y}^i)$$
여기서 $h_f^i$는 반복 횟수 $i$에서의 피드백 특징 맵이다.

**Step 3: 통합 및 재귀적 예측**
다음 단계($i+1$)에서 Forward System의 디코더 $S_d$는 원본 이미지의 특징 $h_s$와 피드백 시스템의 특징 $h_f$를 함께 입력으로 받는다.
$$\hat{y}_{i+1} = S_{d, i+1}(h_s^i, h_f^i)$$
이때 두 특징 공간은 **Concatenation(연결)** 방식으로 병합되어 디코더로 전달된다.

### 3. 학습 절차 (Training Strategy)
두 시스템은 다음과 같은 4단계 반복 학습 과정을 거친다.
1. **Forward System 학습**: 피드백 없이($h_f=0$) 입력 이미지 $x$와 Ground Truth(GT) $y$를 사용하여 가중치 $w_s^i$를 학습한다.
2. **Feedback System 학습**: Forward System의 예측값 $\hat{y}$를 입력으로 하여 GT $y$를 예측하도록 가중치 $w_f^i$를 학습한다.
3. **디코더 정규화 학습**: Forward System의 엔코더와 Feedback System의 엔코더를 고정(Freeze)하고, 두 시스템에서 추출된 특징($h_s, h_f$)을 입력으로 받아 $S_d$의 가중치 $w_{sd}^{i+1}$만을 업데이트한다.
4. **수렴 확인**: 검증 손실(Validation Loss)이 수렴할 때까지 위 과정을 반복한다. (단, 테스트 단계에서는 피드백 시스템의 디코더 부분은 제외하고 엔코더의 특징만을 활용한다.)

### 4. 손실 함수 (Loss Function)
모델은 Binary Cross-Entropy ($L_1$)와 Dice Coefficient Loss ($L_2$)의 평균을 최종 손실 함수로 사용한다.
$$L_{total} = \frac{1}{2} \times (L_1 + L_2)$$
- $L_1$: 픽셀 단위의 분류 정확도를 높이기 위한 Cross-Entropy 손실이다.
- $L_2$: 예측 영역과 실제 영역의 겹침 정도를 최적화하는 Dice 손실이다.

### 5. 네트워크 아키텍처 세부사항
- **Forward System**: U-Net 기반이며, 각 블록은 $3 \times 3$ Convolution $\to$ ELU $\to$ Batch Normalization $\to$ **Squeeze-and-Excitation (SE) Network** 순으로 구성된다. SE-Net은 채널 간의 중요도를 재조정하여 특징 추출 능력을 높인다.
- **Feedback System**: Learning Deconvolution Network 기반의 FCN 구조를 가지며, $3 \times 3$ Convolution, ELU, Batch Normalization을 사용한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 
    - Prostate CT (단일 구조, 저대비 및 금속 아티팩트 포함)
    - Inner Ear $\mu$CT (단일/복잡 구조, 클래스 불균형)
    - Cardiac Cine-MRI (다중 구조, ACDC 데이터셋)
    - Echocardiography (다중 구조, CAMUS 데이터셋)
- **평가 지표**: Dice Similarity Coefficient (DSC), Hausdorff Distance (HD), Relative Volume Difference (RVD).
- **비교 대상**: U-Net, ResU-Net, FCN, Post-DAE, AGN.

### 2. 주요 결과
- **정량적 성과**: 모든 데이터셋에서 LFB-Net이 기존 SOTA 모델들보다 우수한 성능을 보였다. 특히 전립선(Prostate) 분할에서 HD를 U-Net 대비 11mm 감소시켰으며, 심장 MRI의 우심실(RV)과 심근(MYO) 같이 분할이 어려운 구조에서 성능 향상이 두드러졌다.
- **해부학적 타당성**: 정성적 분석 결과, 기존 모델들이 생성하는 결과 내의 '구멍(hole)'이나 비현실적인 분할 영역이 LFB-Net에서는 거의 사라졌으며, 해부학적으로 훨씬 매끄럽고 타당한 결과가 도출되었다.
- **강건성(Robustness)**: 낮은 대비의 이미지나 노이즈가 많은 이미지에서도 결과의 편차가 적었으며, 특히 최악의 케이스(Worst-case)에서의 최대 에러(Max HD)를 유의미하게 낮췄다.

### 3. Ablation Study 결과
- **Feedback Loop의 효과**: 피드백 루프가 없는 Forward System(FS)만 사용했을 때보다 루프를 통합했을 때 DSC와 HD 모두 크게 개선되었다.
- **SE-Block의 영향**: Squeeze-and-Excitation 블록을 제거한 모델(FS*)보다 포함된 모델이 더 높은 정확도를 보였다.
- **병합 전략**: 특징 공간 $h_s$와 $h_f$를 합치는 방법 중 Concatenation이 Addition이나 Multiplication보다 3D HD 지표에서 유의미하게 우수하였다.
- **학습 효율성**: 피드백 루프를 사용했을 때 학습 및 검증 손실의 감소 속도가 더 빨랐으며, 과적합(Overfitting) 경향이 줄어드는 것이 확인되었다. 또한, 파라미터 수가 U-Net(32M)보다 적은 약 8.5M 수준으로 계산 효율성이 높다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석
LFB-Net은 단순히 네트워크를 깊게 쌓는 것이 아니라, **"예측 $\to$ 검토 $\to$ 수정"**이라는 인지적 루프를 아키텍처에 구현하였다. 이는 마치 학생(Forward System)이 시험을 치고 선생님(Feedback System)이 이를 채점하여 다시 알려주는 과정과 유사하다. 이 과정을 통해 모델은 단순한 픽셀 값의 패턴(Texture)을 넘어, 구조적 문맥(Context)과 형태적 특성을 암시적으로 학습하게 된다.

### 2. 한계 및 가정
- **반복 학습의 비용**: 피드백 루프로 인해 학습 시간이 약간 증가(ACDC 데이터셋 기준 약 1시간 추가)한다.
- **2D 위주의 설계**: 본 논문은 3D 데이터셋에 대해서도 2D 기반의 접근법을 적용하여 평가하였다. 저자들은 향후 연구에서 3D 토폴로지를 직접 캡처할 수 있는 3D 버전의 LFB-Net으로 확장을 제안하고 있다.

### 3. 비판적 논의
본 논문은 명시적인 Shape Prior 없이도 유사한 효과를 냈다는 점에서 매우 효율적이다. 하지만 Feedback System이 구체적으로 어떤 '고수준 특징'을 추출하여 Forward System의 오류를 수정하는지에 대한 해석 가능성(Interpretability) 부분은 충분히 다루어지지 않았다. 피드백 맵 $h_f$의 시각적 분석이 추가되었다면 더 설득력 있는 논문이 되었을 것이다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 발생하는 해부학적 불일치 문제를 해결하기 위해, **Forward System과 Feedback System이 상호작용하는 재귀적 루프 구조(LFB-Net)**를 제안한다. 이 모델은 예측 결과를 다시 고차원 특징으로 변환하여 피드백함으로써, 명시적인 형태 정보(Shape Prior) 없이도 해부학적으로 타당하고 견고한 분할 결과를 생성한다. 다양한 임상 데이터셋(CT, MRI, US)에서 SOTA 성능을 입증하였으며, 특히 분할이 어려운 복잡한 구조에서 강점을 보인다. 이는 향후 실시간 의료 영상 분석 및 정밀 진단 시스템에 적용될 가능성이 높다.