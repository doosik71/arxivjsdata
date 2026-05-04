# Causality-inspired Single-source Domain Generalization for Medical Image Segmentation

Cheng Ouyang, Chen Chen, Surui Li, Zeju Li, Chen Qin, Wenjia Bai and Daniel Rueckert (2023)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 의료 영상 분할(Medical Image Segmentation)에서 발생하는 **Domain Shift** 문제, 그 중에서도 단 하나의 소스 도메인 데이터만을 사용하여 학습해야 하는 **Single-source Domain Generalization (DG)** 상황에서의 강건성 확보이다.

의료 영상 분야에서는 촬영 장비, 프로토콜, 제조사 등의 차이로 인해 발생하는 **Acquisition Shift**가 빈번하게 나타난다. 이러한 도메인 간의 차이는 딥러닝 모델이 학습 데이터와 다른 분포를 가진 테스트 데이터(Unseen Domain)에서 성능이 급격히 저하되는 원인이 된다. 특히, 타겟 도메인의 데이터를 미리 확보하기 어려운 의료 데이터의 특성(비용 및 개인정보 보호 문제)으로 인해, 타겟 데이터 없이 소스 도메인만으로 일반화 성능을 높이는 기술이 매우 중요하다.

저자들은 성능 저하의 원인을 두 가지 메커니즘으로 분석한다:
1. **Shifted domain-dependent features**: 영상의 강도(Intensity)와 질감(Texture) 같은 외형적 특성이 도메인마다 다르기 때문에, 모델이 형태(Shape)라는 불변적(Invariant) 특성 대신 외형에 의존하게 된다.
2. **Shifted-correlation effect**: 배경 객체와 관심 객체 간의 가짜 상관관계(Spurious Correlation)가 존재하며, 모델이 이를 단서로 예측을 수행할 경우 타겟 도메인에서 해당 상관관계가 깨지면 성능이 하락한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인과관계(Causality) 관점에서 분석한 Acquisition Shift의 원인을 해결하기 위해, **인과관계 기반의 데이터 증강(Causality-inspired Data Augmentation)** 기법을 제안하는 것이다. 

핵심 기여 사항은 다음과 같다:
- **Global Intensity Non-linear augmentation (GIN)**: 무작위 가중치를 가진 얕은 네트워크를 통해 영상의 강도와 질감을 다양하게 변형함으로써, 모델이 외형이 아닌 도메인 불변 특성인 '형태(Shape)'에 집중하도록 유도한다.
- **Interventional Pseudo-correlation Augmentation (IPA)**: 배경과 관심 객체 간의 가짜 상관관계를 끊어내기 위해, 인과적 개입(Causal Intervention)의 개념을 도입하여 객체들의 외형을 독립적으로 리샘플링하는 기법을 제안한다.
- **포괄적인 검증 환경 구축**: 복부(CT $\rightarrow$ MRI), 심장(bSSFP $\rightarrow$ LGE), 전립선(Cross-center MRI) 등 세 가지 서로 다른 교차 도메인 시나리오에서 제안 방법의 유효성을 검증하였다.

## 📎 Related Works

기존의 도메인 적응 및 일반화 연구들은 다음과 같은 한계를 가진다:
- **Unsupervised Domain Adaptation (UDA)**: 타겟 도메인의 레이블 없는 데이터가 필요하며, 배포 시 타겟 데이터에 맞게 미세 조정(Fine-tuning)해야 하는 번거로움이 있다.
- **Multi-source Domain Generalization (MDG)**: 여러 개의 소스 도메인 데이터가 필요하지만, 실제 의료 현장에서 여러 기관의 데이터를 확보하는 것은 비용과 개인정보 문제로 어렵다.
- **기존 데이터 증강 기법**: Cutout, Mixup, RandConv 등이 제안되었으나, RandConv와 같은 선형 필터링 방식은 실제 의료 영상에서 발생하는 복잡한 도메인 격차를 시뮬레이션하기에는 너무 단순하다는 한계가 있다.

본 논문은 이러한 'Top-down' 방식(타겟 데이터나 다중 소스 데이터에 의존)이 아닌, Acquisition Shift의 인과적 메커니즘에 기반한 'Bottom-up' 방식의 증강 기법을 제안함으로써 차별성을 가진다.

## 🛠️ Methodology

### 1. 문제 정의 및 인과 모델
저자들은 영상 생성 과정을 다음과 같은 인과 그래프로 모델링한다:
- $A \rightarrow X \leftarrow C$: 영상 $X$는 획득 과정(Acquisition, $A$)과 콘텐츠(Content, $C$)라는 두 독립 변수에 의해 생성된다. $C$는 해부학적 형태를, $A$는 강도와 질감을 결정한다.
- $C \rightarrow S \rightarrow Y$: 이상적인 도메인 불변 표현 $S$는 $C$에 의해 결정되며, 이를 통해 정답 마스크 $Y$를 도출할 수 있다.

목표는 $A$의 변화에 관계없이 동일한 예측 결과가 나오는 네트워크 $f_\phi$를 학습하는 것이다. 이를 위해 다음과 같은 손실 함수를 사용한다:
$$\mathcal{L}(\phi) = \mathcal{L}_{seg}(f_\phi(T_i(x)), y) + \mathcal{L}_{seg}(f_\phi(T_j(x)), y) + \lambda_{div} D(p(y|f_\phi(T_i(x))) \| p(y|f_\phi(T_j(x))))$$
여기서 $T_i, T_j$는 서로 다른 광학적 변환(Photometric Transformation)이며, $D$는 KL-Divergence를 통해 두 변환 결과에 대한 예측의 일관성을 강제한다.

### 2. Global Intensity Non-linear augmentation (GIN)
GIN은 영상의 외형을 비선형적으로 변형하여 모델이 외형에 편향되지 않게 한다.
- **구조**: 무작위 가중치 $\theta \sim \mathcal{N}(0, I)$를 가진 얕은 컨볼루션 층과 Leaky ReLU 활성화 함수로 구성된다.
- **작동 방식**: 매 반복(Iteration)마다 새로운 가중치를 샘플링하여 서로 다른 변환 함수 $g_\theta$를 생성한다. 
- **수식**: 원본 영상 $x$와 변환된 영상 $g_{Net_\theta}(x)$를 무작위 계수 $\alpha \in [0, 1]$로 보간한 후, Frobenius norm ($\|\cdot\|_F$)을 사용하여 에너지 수준을 유지한다.
$$g_\theta(x) = \frac{\alpha g_{Net_\theta}(x) + (1-\alpha)x}{\|\alpha g_{Net_\theta}(x) + (1-\alpha)x\|_F} \cdot \|x\|_F$$

### 3. Interventional Pseudo-correlation Augmentation (IPA)
IPA는 배경과 관심 객체 사이의 가짜 상관관계를 제거하기 위해 인과적 개입(do-intervention)을 시뮬레이션한다.
- **핵심 아이디어**: 영상 내의 서로 다른 영역에 서로 다른 외형 변환을 적용함으로써, 객체 간의 외형적 동기화를 깨뜨린다.
- **Pseudo-correlation Map ($b$)**: B-spline 커널을 이용해 무작위 제어 점들을 보간하여 생성한 저주파 스칼라 필드이다. 이 맵은 어떤 픽셀에 어떤 변환을 적용할지 결정하는 가중치 역할을 한다.
- **블렌딩 절차**: 두 개의 서로 다른 GIN 변환 결과 $g_{\theta_1}(x)$와 $g_{\theta_2}(x)$를 맵 $b$를 이용하여 픽셀 단위로 합성한다.
$$T_1(x; \theta_1, \theta_2, b) = g_{\theta_1}(x) \odot b + g_{\theta_2}(x) \odot (1-b)$$
이를 통해 모델은 특정 배경이 나타났을 때 특정 객체가 나타난다는 가짜 상관관계에 의존하지 않고, 객체 자체의 형태에 집중하게 된다.

## 📊 Results

### 실험 설정
- **데이터셋 및 작업**:
    1. 복부 분할 (Abdominal CT $\rightarrow$ T2-SPIR MRI)
    2. 심장 분할 (Cardiac bSSFP $\rightarrow$ LGE MRI)
    3. 전립선 분할 (Cross-center Prostate MRI, 1-versus-5 설정)
- **지표**: Dice Score (0-100)
- **비교 대상**: ERM (기본 학습), Cutout, RSC, MixStyle, AdvBias, RandConv

### 주요 결과
- **정량적 결과**: 제안 방법은 모든 시나리오에서 경쟁 방법들보다 일관되게 높은 Dice Score를 기록하였다. 특히 유사한 접근 방식인 RandConv보다 월등한 성능 향상을 보였는데, 이는 GIN의 비선형 변환과 IPA의 상관관계 제거가 실제 도메인 격차를 더 효과적으로 시뮬레이션함을 시사한다.
- **정성적 결과**: 시각화 결과, 타겟 도메인의 복잡한 영상에서도 제안 방법이 해부학적 구조를 더 정확하게 분할하는 것을 확인하였다.
- **특징 공간 분석**: t-SNE 시각화를 통해, 제안 방법을 사용했을 때 타겟 도메인의 특징들이 소스 도메인의 동일 클래스 특징들과 가깝게 위치하며 클래스 간 구분은 명확함을 확인하였다.

### 절제 실험 (Ablation Study)
- **GIN 설정**: 컨볼루션 층의 수가 너무 적으면 변환 능력이 부족하고, 너무 많으면 현실과 동떨어진 과도한 증강이 일어나 성능이 저하됨을 확인하였다 (4층이 최적).
- **IPA 효과**: IPA를 제거했을 때보다 추가했을 때, 특히 심장과 전립선 데이터셋에서 유의미한 성능 향상이 관찰되었다. 이는 배경-객체 간의 가짜 상관관계를 제거하는 것이 의료 영상 일반화에 필수적임을 입증한다.

## 🧠 Insights & Discussion

본 논문은 단순한 데이터 증강을 넘어, 의료 영상의 도메인 시프트가 발생하는 인과적 메커니즘을 분석하고 이를 제어하기 위한 구체적인 방법론을 제시했다는 점에서 강점이 있다. 특히 타겟 데이터 없이도 강건성을 확보할 수 있다는 점은 실용적 가치가 매우 높다.

**한계 및 논의 사항**:
- **하이퍼파라미터 의존성**: GIN의 층 수나 IPA의 설정 등 일부 하이퍼파라미터를 경험적으로 결정해야 한다는 점이 한계로 지적된다.
- **소스 도메인 성능 저하**: 도메인 불변 특성만을 학습하도록 강제하기 때문에, 소스 도메인 자체에 대한 테스트 성능이 ERM 대비 약간 하락(약 1~2%)하는 Trade-off가 발생한다. 이는 도메인 특수적 특징과 불변적 특징을 적응적으로 조절하는 동적 아키텍처의 필요성을 시사한다.
- **형태적 시프트**: 본 연구는 강도와 질감(Appearance)의 시프트에 집중하였으나, 실제로는 해부학적 형태(Anatomical Shape)의 시프트 또한 존재하므로 이에 대한 후속 연구가 필요하다.

## 📌 TL;DR

이 논문은 단일 소스 데이터만으로 의료 영상 분할 모델의 일반화 성능을 높이기 위해, 인과관계 분석에 기반한 **GIN(비선형 강도 증강)**과 **IPA(가짜 상관관계 제거 증강)** 기법을 제안한다. 제안 방법은 모델이 외형적 특성이나 배경의 가짜 단서에 의존하지 않고 해부학적 '형태'라는 불변 특성을 학습하게 하여, CT-MRI 간 교차 도메인 및 다기관 데이터셋에서 기존 기법보다 뛰어난 강건성을 입증하였다. 이 연구는 타겟 데이터 확보가 어려운 의료 AI 분야에서 실용적인 도메인 일반화 프레임워크를 제공한다.