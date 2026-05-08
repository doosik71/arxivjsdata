# Task-Specific Data Augmentation and Inference Processing for VIPriors Instance Segmentation Challenge

Bo Yan, Xingran Zhao, Yadong Li, Hongbin Wang (2022)

## 🧩 Problem to Solve

본 논문은 데이터가 매우 부족한 환경에서 인스턴스 분할(Instance Segmentation) 성능을 극대화하는 것을 목표로 한다. 구체적으로는 VIPriors Instance Segmentation Challenge의 과제인 농구 코트 이미지 내의 농구공과 인물(선수, 코치, 심판)을 분할하는 문제를 해결하고자 한다.

이 연구의 핵심적인 제약 사항이자 해결해야 할 문제는 다음과 같다.

- **데이터 부족(Data-Deficiency):** 학습 및 검증 데이터셋의 양이 매우 적어 딥러닝 모델이 과적합(Overfitting)되기 쉽다.
- **사전 학습 모델 사용 금지:** 일반적인 컴퓨터 비전 연구와 달리, 외부에서 사전 학습된(Pre-trained) 모델이나 가중치를 사용할 수 없으며 모델을 처음부터(from scratch) 학습시켜야 한다.

따라서 본 논문의 목표는 주어진 도메인의 시각적 귀납 편향(Visual Inductive Priors)을 활용하여, 데이터 부족 문제를 극복하고 높은 분할 정확도를 달성하는 전략을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 도메인 지식을 반영한 **작업 특화 데이터 증강(Task-Specific Data Augmentation, TS-DA)**과 **작업 특화 추론 처리(Task-Specific Inference Processing, TS-IP)** 전략을 설계하는 것이다.

1. **TS-DA:** 농구 코트의 시점(View) 정보와 농구공-사람의 상호작용, 그리고 농구공의 외형적 특성을 반영한 데이터 증강을 통해 데이터 분포의 다양성을 확보한다.
2. **TS-IP:** 농구 코트 이미지의 구성(관중석 위치)과 실제 경기 상황(농구공의 개수)이라는 도메인 제약 조건을 추론 단계에 적용하여 오탐지(False Positive)를 줄이고 정확도를 높인다.

## 📎 Related Works

논문에서는 Mask R-CNN, PANet, TensorMask, CenterMask, SOLO 시리즈 등 기존의 인스턴스 분할 방법론들을 언급한다. 또한, 객체를 한 이미지에서 다른 이미지로 복사해 붙이는 Copy-Paste 증강 기법이 인스턴스 분할에 유용하다는 점을 기반으로 한다.

기존 접근 방식과의 차별점은 일반적인 성능 향상이 아니라, **"사전 학습 모델 없이, 극소량의 데이터만으로"** 특정 도메인(농구)에서 최적의 성능을 내기 위해 도메인 특화적인(Task-Specific) 제약 조건을 증강과 추론 과정에 직접 주입했다는 점이다.

## 🛠️ Methodology

### 1. Task-Specific Data Augmentation (TS-DA)

데이터 부족 문제를 해결하기 위해 두 가지 주요 전략을 사용한다.

#### A. Specific Copy-Paste

단순한 복사-붙여넣기가 아니라 시각적 귀납 편향을 적용한 두 가지 세부 전략을 제안한다.

- **View-Specific Copy-Paste:** 이미지 파일 이름을 통해 왼쪽/오른쪽 시점을 구분하고, 각 시점에 맞는 적절한 위치에 객체를 배치한다.
  - 오른쪽 시점인 경우: $\frac{w}{5} \le x_{min} \le w$
  - 왼쪽 시점인 경우: $0 \le x_{min} \le w - \frac{w}{5}$
  - 공통 높이 제약: $\frac{h}{2} - \frac{h}{5} \le y_{min} \le \frac{h}{2} + \frac{h}{5}$
  - 여기서 $w, h$는 이미지의 너비와 높이이며, $(x_{min}, y_{min})$은 붙여넣을 객체의 좌상단 좌표이다. 이를 통해 객체가 관중석을 제외한 코트 영역에 고르게 분포하도록 한다.

- **Ball-Specific Copy-Paste:** 농구공의 분할 성능을 높이기 위한 전략이다.
  - **Man-Ball Interaction:** 농구공이 사람과 상호작용하는 상황을 모사하기 위해, 선택된 사람의 바운딩 박스 영역 $(s_{x_{min}}, s_{y_{min}}, s_{x_{max}}, s_{y_{max}})$ 내부에 공을 배치한다.
  - **Pure-Ball Generation:** 농구공이 다채로운 색상(Colorful ball)과 단색(Pure ball)으로 나뉜다는 점에 착안하여, RGB 값을 무작위로 변경(예: 갈색 공의 경우 R: 80~90, G/B: 50~60)하여 단색 공 데이터를 인위적으로 생성한다.

#### B. Base-Transform

Copy-Paste 이후에 추가적인 변형을 가한다.

- **Geometric Transform:** 전단(Shear), 회전(Rotate), 이동(Translate) 중 하나를 무작위로 선택하여 적용한다.
- **Photometric Distortion:** 밝기(Brightness), 대비(Contrast), 채도(Saturation), 색조(Hue)를 모두 무작위로 변경한다.

### 2. Task-Specific Inference Processing (TS-IP)

추론 단계에서 도메인 지식을 활용해 결과를 정제한다.

- **Inference Cropping:** 이미지 상단 $1/5$ 영역은 주로 관중석이며 분석 대상이 없다는 점을 이용하여, 이 부분을 잘라내고 모델에 입력한 뒤 결과를 다시 원본 크기로 복원한다.
- **Max Score Filtering:** 한 코트에는 최대 하나의 농구공만 존재한다는 제약을 이용한다.
  - 가장 점수가 높은 농구공 결과 하나만 유지한다.
  - 다른 농구공 후보들의 경우, 최고 점수 결과와 $\text{IoU} > 0$인 경우에만 유지하고 나머지는 제거한다.
  - 또한, 크기가 너무 작거나($<10$) 너무 큰($>40$) 바운딩 박스는 필터링한다.

### 3. Model Architecture 및 학습 절차

- **Model:** $\text{CBSwin-Base}$ 백본과 $\text{CBFPN}$을 사용한 $\text{Hybrid Task Cascade (HTC)}$ 디텍터를 기반으로 한다. 여기에 Mask 품질을 직접 학습하는 $\text{MaskIoU head}$ (Mask Scoring R-CNN)를 추가하여 성능을 높였다.
- **Training:**
  - $\text{AdamW}$ 옵티마이저를 사용하며 초기 학습률은 $0.0001$이다.
  - 모델이 수렴한 후, 모델의 강건성(Robustness)을 높이기 위해 $\text{SWA (Stochastic Weight Averaging)}$ 기법을 사용하여 파인튜닝한다.
  - 학습 데이터와 검증 데이터를 10배 복제하여 사용하며, 숏사이드 기준 $820 \sim 3080$ 픽셀로 무작위 스케일링 후 $(1920, 1440)$ 크기로 크롭 및 패딩하여 입력한다.

## 📊 Results

### 실험 설정

- **데이터셋:** VIPriors Instance Segmentation Challenge 제공 데이터 (학습 1840장, 검증 620장).
- **평가 지표:** $\text{AP@0.50:0.95}$.
- **기준선 (Baseline):** $\text{HTC-CBSwinBase}$ 모델.

### 정량적 결과

최종 제안 방법론은 테스트 세트에서 **$0.531 \text{ AP@0.50:0.95}$**를 달성하였다.

### Ablation Study (성능 향상 단계)

각 구성 요소가 성능에 기여한 정도는 다음과 같다 (mAP 기준).

- **Baseline (HTC-CBSwinBase):** $0.396$
- **+ MaskIoU head:** $0.415$ ($\uparrow 0.019$)
- **+ TS-DA (데이터 증강):** $0.472$ ($\uparrow 0.057$)
- **+ TTA (Horizontal Flip, Multi-scale):** $0.477$ ($\uparrow 0.005$)
- **+ Mask Threshold Binary (0.5 $\to$ 0.4):** $0.489$ ($\uparrow 0.012$)
- **+ SWA Training:** $0.512$ ($\uparrow 0.023$)
- **+ Inference Cropping:** $0.522$ ($\uparrow 0.010$)
- **+ Max Score Filtering:** $0.529$ ($\uparrow 0.007$)
- **+ Ensemble (Mask Loss Weight=2.0):** $0.531$ ($\uparrow 0.002$)

## 🧠 Insights & Discussion

### 강점

본 논문은 데이터가 극도로 부족한 상황에서 단순한 일반 목적의 증강 기법에 의존하지 않고, **"농구 경기"라는 특정 도메인의 시각적 특성(코트 구조, 공의 개수, 색상, 인물과의 위치 관계)**을 수학적 제약 조건으로 변환하여 모델에 주입했다는 점에서 매우 실용적인 접근법을 보여준다. 특히 TS-DA가 가장 큰 성능 향상($+0.057$)을 이끌어낸 점은 데이터 부족 문제에서 도메인 특화 증강이 얼마나 중요한지를 시사한다.

### 한계 및 논의사항

- **도메인 의존성:** 제안된 방법론은 농구 코트라는 매우 특수한 환경에 최적화되어 있다. 만약 다른 스포츠나 일반적인 환경으로 확장한다면, 각 도메인에 맞는 새로운 '귀납 편향(Inductive Priors)'을 다시 정의해야 하는 번거로움이 있다.
- **추론 단계의 제약:** $\text{Max Score Filtering}$에서 공이 하나만 존재한다고 가정한 점은 실제 경기 중 공이 여러 개 투입되는 특수한 상황에서는 오작동할 가능성이 있다.
- **SWA의 효과:** 사전 학습 모델을 사용할 수 없는 상황에서 $\text{SWA}$를 통한 파인튜닝이 상당한 성능 향상($+0.023$)을 가져온 점은, 데이터가 적을수록 최적의 가중치 지점을 찾는 최적화 전략이 중요함을 보여준다.

## 📌 TL;DR

본 논문은 사전 학습 모델 사용이 금지된 데이터 부족 환경의 농구 인스턴스 분할 문제를 해결하기 위해, **도메인 지식을 반영한 데이터 증강(TS-DA)**과 **추론 후처리(TS-IP)** 전략을 제안하였다. 농구 코트의 시점, 공의 특성, 배치 제약 등을 활용하여 최종적으로 $0.531 \text{ AP}$를 달성하였으며, 이는 도메인 특화 지식이 데이터 부족 문제를 보완하는 강력한 수단이 될 수 있음을 입증한다.
