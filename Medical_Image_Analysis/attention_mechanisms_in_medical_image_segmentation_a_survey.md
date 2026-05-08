# Attention Mechanisms in Medical Image Segmentation: A Survey

Yutong Xie, Bing Yang, Qingbiao Guan, Jianpeng Zhang, Qi Wu, Yong Xia (2023)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 컴퓨터 보조 진단(Computer-Aided Diagnosis, CAD)의 핵심적인 요소이다. 하지만 의료 영상의 특성상 다음과 같은 세 가지 주요 난제가 존재한다. 첫째, 연부 조직의 낮은 대비(Low soft tissue contrast)로 인해 객체의 경계가 모호하다. 둘째, 해부학적 또는 병리학적 구조의 모양, 크기, 위치가 매우 다양하다. 셋째, 전문가의 노동력과 전문 지식이 필요하므로 학습에 필요한 충분한 양의 어노테이션된 데이터를 확보하기 어렵다.

이러한 문제는 모델이 객체와 배경 사이의 세만틱 관계를 적절히 모델링하는 것을 어렵게 만든다. 본 논문의 목표는 인간의 시각 인지 시스템이 관심 영역에 집중하고 무관한 배경 정보를 무시하는 방식을 모방한 'Attention Mechanism'이 의료 영상 분할에서 어떻게 적용되고 있는지 체계적으로 분석하고 분류하는 것이다.

## ✨ Key Contributions

본 논문은 300편 이상의 의료 영상 분할 관련 논문을 분석하여 다음과 같은 핵심적인 기여를 한다.

1. **체계적인 분류 체계(Taxonomy) 제시**: Attention 기법을 'Non-Transformer Attention'과 'Transformer Attention'의 두 그룹으로 대분류하고, 각 그룹을 다시 **What to use**(어떤 메커니즘을 사용하는가), **How to use**(네트워크의 어디에 배치하는가), **Where to use**(어떤 임상 작업에 적용하는가)의 세 가지 관점에서 심층 분석하였다.
2. **포괄적인 리뷰**: 단순한 Transformer 기반 연구뿐만 아니라 전통적인 Attention 기법까지 모두 아우름으로써, 의료 영상 분할 분야의 전반적인 연구 맥락을 제공한다.
3. **미래 연구 방향 제시**: 작업 특이성(Task specificity), 강건성(Robustness), 표준 평가 지표의 부재, 다중 모달리티 및 다중 작업 학습, 계산 복잡도 등 향후 해결해야 할 도전 과제들을 명시적으로 논의하였다.

## 📎 Related Works

기존의 서베이 논문들(Shamshad et al., He et al. 등)은 주로 의료 영상 분석 전반에서 Transformer의 적용 사례에 집중하였다. 반면, 본 논문은 다음과 같은 차별점을 가진다.

- **작업의 구체성**: 의료 영상 분석의 여러 작업 중 특히 난이도가 높고 중요도가 큰 '분할(Segmentation)' 작업에 집중하여 더 깊은 수준의 분석을 제공한다.
- **범위의 확장성**: Transformer 기반의 방법론뿐만 아니라, CNN의 플러그인 형태로 사용되는 전통적인 Attention 메커니즘까지 모두 포함하여 분석하였다.

## 🛠️ Methodology

본 논문은 서베이 논문이므로 새로운 알고리즘을 제안하는 대신, 기존 방법론들을 분석하기 위한 프레임워크를 제시한다.

### 1. Non-Transformer Attention

전통적인 Attention은 크게 Channel, Spatial, Temporal Attention으로 구분된다. 일반적인 수식은 다음과 같다.
$$\text{Attention} = f(g(x), x)$$
여기서 $g(x)$는 생성된 Attention 맵이며, $f(g(x), x)$는 이 맵을 기반으로 입력 벡터 $x$를 처리하는 과정을 의미한다.

- **Channel Attention**: 각 채널이 서로 다른 객체를 대표한다고 가정하고, 채널 간의 관계를 캡처하여 가중치를 재조정한다 (예: SE-Net).
- **Spatial Attention**: 특징 맵의 공간적 영역에 중요도 점수를 부여하여 중요한 영역을 식별한다 (예: Attention Gates).
- **Temporal Attention**: 비디오와 같이 시간축이 존재하는 데이터에서 동적인 프레임 선택 메커니즘으로 작용한다.

### 2. Transformer Attention

Transformer의 핵심은 Multi-Head Self-Attention (MHSA)이다. 입력 벡터를 Query($Q$), Key($K$), Value($V$)로 변환하여 다음과 같이 계산한다.
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Multi-Head Attention은 이를 여러 개의 헤드로 확장하여 서로 다른 표현 하위 공간에서 관계를 캡처한 뒤 결합(Concat)하고 선형 투영($W^O$)을 수행한다.

### 3. 네트워크 내 배치 전략 (How to Use)

논문은 Attention이 적용되는 위치를 다음과 같이 분류한다.

- **Encoder**: 수용 영역(Receptive Field)을 확장하고 더 풍부한 인코딩 정보를 추출하기 위해 사용한다. (Bottleneck, 각 단계별, 혹은 두 네트워크 사이의 전송 경로에 배치)
- **Decoder**: 잠재 표현을 고해상도 세그멘테이션 맵으로 복원할 때 세밀한 특징을 생성하기 위해 사용한다.
- **Skip Connection**: 인코더와 디코더 사이의 세만틱 갭(Semantic Gap)을 줄이고, 정보 전달 과정에서 노이즈를 억제하며 유용한 특징을 선택적으로 증폭시킨다.
- **Hybrid**: 위의 여러 위치에 서로 다른 Attention 모듈을 복합적으로 배치하는 전략이다.

## 📊 Results

본 논문은 방대한 양의 실험 결과를 표(Table 2~16) 형태로 정리하여 제시한다.

### 실험 설정 및 지표

- **대상 데이터셋**: BraTS (뇌종양), BCV (다중 장기), ACDC (심장), ISIC (피부 병변) 등 다양한 의료 데이터셋을 다룬다.
- **평가 지표**: Dice Coefficient, Jaccard Index (JI), Accuracy (ACC), Hausdorff Distance (HD), Area Under Curve (AUC) 등이 사용되었다.

### 주요 분석 결과

1. **비-Transformer 계열**: 피부 병변, 전립선, 폴립 분할 등 경계가 모호한 작업에서는 **Spatial Attention**이 Channel Attention보다 더 선호된다. 이는 경계 정보(Edge information)가 분할 성능에 결정적인 영향을 미치기 때문이다.
2. **Transformer 계열**:
    - **2D 데이터셋**에서는 Hybrid Encoder + CNN Decoder 구조가 효율적인 성능을 보인다.
    - **3D 데이터셋**에서는 Transformer Encoder + Transformer Decoder 구조가 더 우수한 성능을 나타내는 경향이 있다.
3. **보편적 모델**: BCV 데이터셋을 이용한 다중 장기 분할 실험 결과, 최신 Transformer 기반 모델들이 기존 CNN 기반 모델들보다 높은 Dice score를 기록하며 전역적 문맥 파악 능력이 뛰어남을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 Attention 메커니즘이 단순한 성능 향상을 넘어, 모델이 입력 데이터의 어느 부분에 집중하고 있는지를 보여줌으로써 딥러닝의 '블랙박스' 문제를 일부 해결하는 **해석 가능성(Interpretability)**을 제공한다는 점을 강조한다.

### 한계 및 비판적 해석

1. **작업 특이적 설계의 부족**: 분석 결과, 대부분의 Attention 모듈이 일반적인 구조를 그대로 가져다 쓴 '이식(Transplant)' 수준에 머물러 있다. 특정 장기나 질환의 해부학적 특성을 반영한 **Task-specific Attention** 설계가 부족하다는 점은 큰 한계로 지적된다.
2. **데이터 효율성 문제**: Transformer는 인덕티브 바이어스(Inductive Bias)가 부족하여 방대한 양의 데이터를 요구한다. 하지만 의료 데이터는 희소하므로, 대부분의 연구가 ImageNet 사전 학습(Pre-training)에 의존하고 있어 실제 의료 도메인으로의 최적화가 더 필요하다.
3. **비교의 불명확성**: 논문마다 사용하는 데이터셋의 분할 방식, 전처리 과정, 평가 지표가 상이하여 모델 간의 정량적 성능 비교가 불공정하게 이루어지고 있다.

## 📌 TL;DR

본 논문은 의료 영상 분할 분야에서 사용된 300편 이상의 Attention 기반 연구를 **What, How, Where**라는 세 가지 관점에서 체계적으로 분류한 종합 서베이 보고서이다. 전통적인 CNN 기반 Attention부터 최신 Transformer 구조까지 망라하였으며, 특히 공간적 주의 집중(Spatial Attention)의 중요성과 Transformer의 전역적 문맥 모델링 능력을 확인하였다. 향후 연구는 단순한 구조 도입을 넘어 의료 영상의 특성을 반영한 **작업 특화형 Attention 설계**와 **계산 복잡도 최적화**, 그리고 **표준화된 평가 벤치마크 구축** 방향으로 나아가야 함을 시사한다.
