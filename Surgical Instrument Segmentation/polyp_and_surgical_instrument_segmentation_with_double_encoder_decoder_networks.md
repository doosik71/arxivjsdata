# POLYP AND SURGICAL INSTRUMENT SEGMENTATION WITH DOUBLE ENCODER-DECODER NETWORKS

Adrian Galdran(2024)

## 🧩 Problem to Solve

본 논문은 내시경 이미지에서 용종(Polyps)과 수술 도구(Surgical Instruments)를 정밀하게 분할(Segmentation)하는 문제를 해결하고자 한다. 대장암과 같은 질병의 조기 발견 및 치료를 위해서는 대장 내시경 검사가 필수적이며, 이때 AI 기반의 자동 분석 및 의사 결정 지원 시스템은 검사의 효율성을 높이고 의료진의 개입을 용이하게 하는 데 중요한 역할을 한다.

본 연구의 구체적인 목표는 MedAI 챌린지의 요구 사항에 맞추어, 서로 다른 특성을 가진 용종과 수술 도구라는 두 가지 독립적인 세그멘테이션 작업을 수행하는 고성능 딥러닝 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 두 개의 인코더-디코더 네트워크를 순차적으로 결합한 Double Encoder-Decoder 구조를 사용하는 것이다. 첫 번째 네트워크가 생성한 출력물을 일종의 Relevance map으로 활용하여, 두 번째 네트워크가 이미지의 가장 중요한 영역에 집중할 수 있도록 설계하였다.

또한, 이전 연구의 성과를 바탕으로 다음과 같은 세 가지 주요 개선 사항을 도입하였다.

1. 더 강력한 성능의 인코더 아키텍처(Resnext101)와 디코더(FPN) 채택
2. SAM(Sharpness-Aware Minimization)을 활용한 최적화 절차 개선
3. Temperature sharpening 기반의 모델 앙상블 포스트 프로세싱을 통한 정밀도-재현율(Precision-Recall) 최적화

## 📎 Related Works

일반적으로 시맨틱 세그멘테이션은 특징 추출을 위한 인코더와 원래 해상도로 복원하는 디코더가 결합된 Encoder-Decoder 네트워크(예: U-Net)를 통해 수행된다. 본 논문은 여기서 더 나아가 두 개의 네트워크를 직렬로 연결한 Double Encoder-Decoder 구조를 사용한다.

저자는 이전에 EndoTect 2021 챌린지에서 우승했던 자신의 연구를 기반으로 하며, 해당 연구에서 사용된 구조를 확장하여 이번 MedAI 경쟁의 두 가지 작업(용종 및 도구 분할)에 적용하였다. 기존 접근 방식과의 차별점은 단순한 아키텍처 적용에 그치지 않고, 최적화 알고리즘(SAM)과 앙상블 기법(Temperature scaling)을 통해 실질적인 성능을 극대화했다는 점에 있다.

## 🛠️ Methodology

### 1. 시스템 구조: Double Encoder-Decoder

전체 시스템은 두 개의 인코더-디코더 네트워크 $E^{(1)}$과 $E^{(2)}$로 구성된다. 입력 RGB 이미지 $x$가 첫 번째 네트워크 $E^{(1)}$을 통과하면 픽셀 단위의 확률 맵이 생성된다. 이후 이 출력값 $E^{(1)}(x)$는 원본 이미지 $x$와 채널 방향으로 결합(stack)되어 두 번째 네트워크 $E^{(2)}$의 입력으로 들어간다. 결과적으로 $E^{(2)}$는 4채널 입력(RGB 3채널 + $E^{(1)}$의 출력 1채널)을 받아 최종 세그멘테이션 결과를 도출한다. 이를 수식으로 나타내면 다음과 같다.

$$E(x) = E^{(2)}(x, E^{(1)}(x))$$

### 2. 주요 구성 요소

- **Encoder**: 사전 학습된 Resnext101 모델을 사용하여 강력한 특징 추출 능력을 확보하였다.
- **Decoder**: Feature Pyramid Network (FPN) 아키텍처를 사용하여 다양한 스케일의 특징을 효과적으로 통합하였다.
- **Optimization**: ADAM 옵티마이저를 SAM(Sharpness-Aware Minimization)으로 감싸서 일반화 성능을 향상시켰다.

### 3. 학습 및 추론 절차

- **데이터 증강 및 앙상블**: 훈련 및 검증 데이터를 4-fold 방식으로 회전시켜 4개의 서로 다른 모델 $E_1, E_2, E_3, E_4$를 학습시켰다.
- **Temperature Sharpening**: 최종 예측 시, 앙상블된 모델들의 결과에 온도 파라미터 $t$를 적용하여 다음과 같이 결합하였다.

$$p = \frac{1}{4} \sum_{i=1}^{4} (E_i(x)^t)$$

여기서 $t$는 자유 파라미터이며, 이 값을 조절함으로써 최종 예측의 확신도를 제어하고 Precision과 Recall 사이의 트레이드오프를 조정할 수 있다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 용종 분할을 위해 Kvasir-SEG를, 수술 도구 분할을 위해 Kvasir-Instrument 데이터셋을 사용하였다. 성능 평가는 MedAI 주최 측이 보유한 숨겨진 테스트 세트(Hidden test set)에서 수행되었다.
- **평가 지표**: Dice score, Precision, Recall 세 가지 지표를 사용하였다. 특히 Dice score는 다음과 같이 정의된다.

$$D(f_\theta(x), y) = \frac{2 \cdot |f_\theta(x) \cap y|}{|f_\theta(x)| + |y|} = \frac{2TP}{2TP + FP + FN}$$

### 2. 주요 결과

실험 결과, 두 작업 모두에서 Dice score 90% 내외의 높은 성능을 보였다. 온도 파라미터 $t$에 따른 성능 변화는 다음과 같다.

- **용종 분할 (Polyps)**: $t=1$일 때 Dice score가 89.65%로 가장 높았으며, $t$ 값이 낮아질수록 Recall이 증가하고 Precision과 Dice score가 감소하는 경향을 보였다.
- **수술 도구 분할 (Instruments)**: $t=1$일 때 Dice score 96.18%로 매우 높은 성능을 기록하였다. 용종 분할보다 전반적으로 성능이 더 높게 나타났다.

| Temperature | Task | Dice | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| $T=0.5$ | Polyps | 88.59 | 87.76 | 94.56 |
| $T=1.0$ | Polyps | 89.65 | 92.42 | 90.09 |
| $T=2.0$ | Polyps | 88.69 | 93.37 | 87.56 |
| $T=0.5$ | Instruments | 94.94 | 92.08 | 98.84 |
| $T=1.0$ | Instruments | 96.18 | 95.06 | 97.88 |
| $T=2.0$ | Instruments | 96.35 | 96.32 | 96.92 |

## 🧠 Insights & Discussion

본 연구를 통해 도출된 주요 인사이트는 다음과 같다.
첫째, 수술 도구 세그멘테이션이 용종 세그멘테이션보다 상대적으로 쉬운 작업임을 확인하였다. 이는 용종이 색상이나 외형 면에서 훨씬 더 큰 변동성을 가지기 때문인 것으로 분석된다.
둘째, Temperature sharpening이 세그멘테이션 작업에서 Precision과 Recall의 트레이드오프를 제어하는 유효한 메커니즘임을 입증하였다. 낮은 온도는 Recall을 높이는 반면, 높은 온도는 Precision과 Dice score를 높이는 경향이 있다.

한계점으로는 학습 데이터에는 항상 대상 객체(용종 또는 도구)가 포함되어 있었으나, 실제 테스트 세트에는 객체가 없는 프레임이 존재했다는 점이 언급되었다. 이는 Out-of-distribution(OOD) 데이터 문제로, 향후 연구에서는 이러한 사례를 더 잘 처리하기 위한 방법론에 대한 조사가 필요하다.

## 📌 TL;DR

본 논문은 Double Encoder-Decoder 구조에 Resnext101 인코더, FPN 디코더, SAM 옵티마이저를 결합하여 내시경 이미지 내 용종과 수술 도구를 정밀하게 분할하는 방법론을 제안한다. 특히 Temperature sharpening 기반의 앙상블 기법을 통해 정밀도와 재현율을 최적화하였으며, 이를 통해 두 작업 모두에서 90% 이상의 Dice score를 달성하였다. 이 연구는 의료 영상 분석에서 모델의 확신도를 조절하여 성능을 최적화하는 실용적인 접근법을 제시했다는 점에서 향후 연구 및 실무 적용에 기여할 가능성이 크다.
