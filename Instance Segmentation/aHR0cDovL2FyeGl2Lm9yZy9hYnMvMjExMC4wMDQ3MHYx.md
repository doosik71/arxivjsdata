# Task-Specific Copy-Paste Data Augmentation Method for Instance Segmentation

Jahongir Yunusov, Shohruh Rakhmatov, Abdulaziz Namozov, Abdulaziz Gaybulayev, Tae-Hyong Kim (2021)

## 🧩 Problem to Solve

본 연구는 데이터가 매우 부족한 환경에서 Instance Segmentation 모델의 성능을 높이는 것을 목표로 한다. 일반적으로 딥러닝 기반의 Instance Segmentation은 높은 성능을 달성하기 위해 대규모 학습 데이터셋이 필요하며, 특히 픽셀 단위의 정교한 어노테이션(Annotation) 작업은 시간과 비용 소모가 매우 크다.

본 논문은 ICCV 2021의 VIPriors 워크숍 챌린지에 참여하며 직면한 제약 사항을 해결하고자 한다. 해당 챌린지는 데이터가 극도로 부족한 설정에서 모델을 처음부터(from scratch) 학습시켜야 하며, 제공된 학습 데이터 외에 외부 데이터셋 사용, 사전 학습(Pre-training), 전이 학습(Transfer Learning)이 모두 금지된 매우 까다로운 조건이다. 구체적으로 학습 데이터는 단 184장의 농구 경기 장면 이미지로 구성되어 있으며, 타겟 클래스는 선수(Player)와 공(Ball) 두 가지이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 일반적인 Copy-Paste 데이터 증강 기법을 해당 도메인(농구 경기)의 특성에 맞게 변형한 **Task-Specific Copy-Paste** 방법론을 제안하는 것이다.

기존의 Copy-Paste 방식은 객체를 이미지 내의 임의의 위치에 붙여넣는 방식이지만, 본 연구에서는 객체가 실제로 존재할 가능성이 높은 특정 영역으로 붙여넣기 위치를 제한함으로써 모델의 일반화 성능을 높이고 학습 효율을 개선하였다. 또한, RandAugment와 GridMask와 같은 추가적인 증강 기법과 고성능 백본 네트워크(CBSwin-B) 및 하이퍼파라미터 최적화를 결합하여 데이터 부족 문제를 극복하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급한다.

- **Data Augmentation**: 색상 공간 변환(Color-space transformation) 및 기하학적 변환(Geometric transformation)과 같은 일반적인 기법부터, 두 이미지를 임의의 비율로 섞는 Mixup, 그리고 한 이미지의 객체를 다른 이미지로 복사하는 Copy-Paste 기법이 언급된다. 특히 Copy-Paste는 Instance Segmentation에서 매우 효과적이라고 설명한다.
- **Backbone Network**: CNN 기반의 ResNet, Res2Net부터 최근 NLP에서 전이되어 높은 성능을 보이는 Transformer 기반의 Swin Transformer가 언급된다. 특히 고수준 및 저수준 특징을 결합하여 효율성을 높인 CBNetV2가 최신 SOTA 모델로 소개된다.
- **Instance Segmentation**: 객체를 검출하고 각 인스턴스의 픽셀을 분리하는 작업으로, 본 연구에서는 특수한 캐스케이드 구조를 가진 Hybrid Task Cascade (HTC) 모델을 사용한다.

## 🛠️ Methodology

### 1. Task-Specific Copy-Paste
데이터 증강 프로세스는 다음과 같은 단계로 진행된다.
1. 모든 이미지에서 객체를 크롭(Crop)하여 해당 이미지와 마스크 어노테이션을 저장한다.
2. 원본 이미지들을 20배로 복제하여 증강 기법의 입력값으로 사용한다.
3. 클래스 불균형을 방지하기 위해 각 클래스에서 5개에서 15개 사이의 인스턴스를 무작위로 샘플링한다.
4. **위치 제약 조건 적용**: 객체를 무작위로 배치하는 대신, 농구 경기 장면의 특성을 고려하여 다음과 같은 좌표 범위 내에만 붙여넣는다.

$$256 \le x_{min} \le w - 256$$
$$\frac{w}{2} + 256 \le y_{min} \le \frac{h}{2} + 256$$

여기서 $w$와 $h$는 이미지의 너비와 높이이며, $(x_{min}, y_{min})$은 붙여넣을 객체의 좌측 상단 좌표를 의미한다. (참고: 수식 (2)에서 $w$와 $h$가 혼용된 것으로 보이나, 원문 텍스트에 기재된 내용을 그대로 기술한다.)

### 2. 모델 아키텍처 및 학습 설정
- **모델 구조**: CBSwin-B 백본과 CBFPN을 결합한 HTC (Hybrid Task Cascade) 검출기를 사용한다.
- **활성화 함수**: Box 및 Mask Head에 사용된 모든 ReLU 함수를 SiLU (Sigmoid Linear Unit)로 교체하였다.
- **학습 절차**: 
    - Multi-scale 모드에서 Random Sampler를 사용하여 학습하며, 학습 스케줄은 6x schedule을 적용하였다.
    - 입력 이미지의 짧은 쪽 길이를 800에서 1400 사이로 랜덤하게 스케일링하였다.
- **추론 절차**: 테스트 단계에서는 TTA (Test Time Augmentation) 없이 $(1600, 1400)$ 단일 스케일 모드로 평가하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: VIPriors 챌린지 제공 데이터 (Train 184장, Val 62장, Test 64장)
- **평가 지표**: $AP@0.50:0.95$
- **베이스라인**: CBSwin-T 백본, CBFPN, 2x schedule을 사용한 HTC detector

### 정량적 결과
실험 단계별 성능 향상 결과는 다음과 같다.

| 방법론 (Methods) | 스케줄 | Val $AP@0.50:0.95$ | Test $AP@0.50:0.95$ |
| :--- | :---: | :---: | :---: |
| Baseline | 2x | 0.186 | - |
| + TS Copy-Paste | 2x | 0.338 | - |
| + RandAugment + GridMask | 2x | 0.345 | - |
| + Better backbone (CBSwin-B) | 6x | 0.398 | 0.433 |
| + Val set added in training | 6x | - | 0.477 |

### 결과 분석
Task-Specific Copy-Paste를 적용했을 때 성능이 가장 비약적으로 상승($0.186 \rightarrow 0.338$)하였으며, 백본을 CBSwin-B로 교체하고 학습 스케줄을 늘렸을 때 추가적인 성능 향상이 있었다. 최종적으로 검증 세트(Validation set)를 학습 데이터에 포함시켜 학습했을 때 테스트 세트에서 최고 성능인 $0.477$ $AP$를 달성하였다.

## 🧠 Insights & Discussion

본 논문은 데이터가 매우 부족한 특수한 상황에서 도메인 지식을 활용한 데이터 증강이 얼마나 효과적인지를 보여준다. 특히, 단순한 무작위 증강보다 객체가 나타날 법한 위치를 제한한 'Task-Specific' 접근 방식이 모델의 수렴과 일반화에 긍정적인 영향을 주었음을 알 수 있다.

다만, 본 보고서에서 제시한 수식 (2)의 좌표 제한 범위($\frac{w}{2} + 256 \le y_{min} \le \frac{h}{2} + 256$)는 일반적인 이미지 좌표계 관점에서 볼 때 수식상의 오류(변수 $w, h$의 혼용 또는 범위 설정의 모순)가 있을 가능성이 높으나, 이에 대한 상세한 설명이 부족하다. 또한, 검증 세트를 학습에 포함시킨 결과는 성능 수치를 높이는 데 기여했으나, 이는 엄격한 의미의 일반화 성능 측정이라기보다 주어진 데이터셋을 최대한 활용한 결과로 해석된다.

## 📌 TL;DR

본 연구는 데이터가 극도로 부족한 농구 경기 영상의 Instance Segmentation 문제에서, 객체의 배치 위치를 특정 영역으로 제한한 **Task-Specific Copy-Paste** 증강 기법을 제안하였다. 이를 CBSwin-B 백본 및 HTC 구조와 결합하여, 외부 데이터나 사전 학습 없이도 $0.477$ $AP$라는 성과를 거두었으며, 이는 도메인 특화 데이터 증강이 데이터 부족 문제를 해결하는 강력한 도구가 될 수 있음을 시사한다.