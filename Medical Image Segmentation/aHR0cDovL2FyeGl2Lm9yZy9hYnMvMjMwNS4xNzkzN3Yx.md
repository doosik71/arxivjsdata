# Attention Mechanisms in Medical Image Segmentation: A Survey

Yutong Xie, Bing Yang, Qingbiao Guan, Jianpeng Zhang, Qi Wu, Yong Xia (2023)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 컴퓨터 보조 진단(CAD)의 핵심적인 요소이지만, 다음과 같은 세 가지 주요 기술적 어려움이 존재한다. 첫째, 연조직 간의 낮은 대비(low soft tissue contrast)로 인해 객체의 경계가 모호하다. 둘째, 해부학적 또는 병리학적 구조의 모양, 크기, 위치가 환자마다 매우 다양하다. 셋째, 전문 지식과 높은 비용으로 인해 학습에 필요한 충분한 양의 어노테이션 데이터(annotated images)를 확보하기 어렵다.

이러한 문제는 모델이 객체와 배경 사이의 세맨틱 관계(semantic relationship)를 정확하게 모델링하는 것을 방해한다. 본 논문의 목표는 인간의 시각 인지 시스템이 관심 영역에 집중하고 무관한 배경 정보를 무시하는 것처럼, 신경망이 타겟 작업과 관련된 중요 영역에 적응적으로 가중치를 부여하는 Attention Mechanism의 원리와 의료 영상 분할 분야에서의 적용 사례를 체계적으로 분석하고 분류하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 300편 이상의 관련 문헌을 조사하여 의료 영상 분할에서의 Attention Mechanism을 체계적으로 분류한 Taxonomy를 제시한 점이다. 특히, 단순한 나열이 아니라 다음과 같은 세 가지 관점에서 분석을 수행하였다.

1.  **What to use (원리):** 어떠한 Attention 기법(Channel, Spatial, Temporal, Transformer 등)이 사용되었는가.
2.  **How to use (구현 방법):** 네트워크의 어느 위치(Encoder, Decoder, Skip Connection 등)에 Attention 레이어가 삽입되었는가.
3.  **Where to use (적용 대상):** 어떤 임상 작업(뇌, 심장, 간, 폐 등 특정 장기나 병변 분할)에 적용되었는가.

또한, 기존의 Non-Transformer 기반 Attention과 최근 주류가 된 Transformer 기반 Attention을 명확히 구분하여, 각각의 특성과 한계점을 심도 있게 분석하였다.

## 📎 Related Works

기존에도 의료 영상 분석 전반이나 Transformer 기반의 응용 연구에 대한 서베이 논문들이 존재하였다. 하지만 본 논문은 다음과 같은 차별점을 가진다.

-   **범위의 포괄성:** Transformer 기반 방법론에만 국한되지 않고, 전통적인 Non-Transformer 기반 Attention 기법까지 모두 아우른다.
-   **작업의 특수성:** 일반적인 의료 영상 분석(분류, 검출 등)이 아닌 '분할(Segmentation)' 작업에 집중하여, 해당 작업에서 발생하는 구체적인 문제(경계 모호성 등)와 Attention의 관계를 더 깊게 분석하였다.

## 🛠️ Methodology

본 논문은 Attention Mechanism을 크게 두 가지 범주로 나누어 설명한다.

### 1. Non-Transformer Attention

전통적인 Attention은 주로 CNN의 플러그인 형태로 사용되며, 기본 수식은 다음과 같이 정의된다.
$$\text{Attention} = f(g(x), x)$$
여기서 $g(x)$는 생성된 Attention 맵이며, $f$는 이를 이용해 입력 벡터 $x$를 처리하는 과정을 의미한다.

#### 주요 유형 (What to use)
-   **Channel Attention:** 각 채널을 서로 다른 객체로 간주하고 가중치를 재조정한다. 대표적으로 Squeeze-and-Excitation (SE) 블록이 있으며, Global Average Pooling을 통해 채널 간 관계를 캡처한다.
-   **Spatial Attention:** 특성 맵의 공간적 영역에 중요도 점수를 부여하여 중요 영역을 식별한다. Attention Gates나 Non-local networks가 대표적이다.
-   **Temporal Attention:** 비디오와 같은 시계열 데이터에서 동적인 타임 프레임을 선택하는 메커니즘이다.

#### 구현 위치 (How to use)
-   **Encoder:** Receptive field를 확장하고 더 풍부한 인코딩 정보를 추출하기 위해 Bottleneck이나 각 스테이지에 삽입한다.
-   **Decoder:** 고해상도 분할 맵으로 복원하는 과정에서 배경 소음을 제거하고 에지(edge) 정보를 보존하기 위해 사용한다.
-   **Skip Connection:** Encoder와 Decoder 사이의 세맨틱 갭(semantic gap)을 줄여 정보 전달을 최적화한다. (예: Attention U-Net의 Attention Gate)

### 2. Transformer Attention

Transformer는 전역적 문맥 의존성(global contextual dependency)을 모델링하는 데 탁월하며, 핵심은 Scaled Dot-Product Attention이다.

#### 주요 방정식
입력 벡터 $x$가 Query($Q$), Key($K$), Value($V$)로 변환될 때, Attention 연산은 다음과 같다.
$$g(x) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$
$$f(g(x), x) = g(x)V$$

다중 헤드 주의 집중(Multi-Head Attention)은 서로 다른 표현 공간에서 정보를 캡처하며 다음과 같이 정의된다.
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

#### 네트워크 아키텍처 (How to use)
-   **Hybrid Encoder + CNN Decoder:** CNN의 지역성(locality)과 Transformer의 전역적 모델링 능력을 결합한 형태이다. (예: TransUNet)
-   **Pure Transformer Encoder + CNN Decoder:** Encoder를 ViT 등으로 대체하여 계층적 특징을 추출한다. (예: UNETR)
-   **Transformer Encoder + Transformer Decoder:** 전체 구조를 Transformer 블록으로 구성하여 전역적 일관성을 극대화한다. (예: Swin-UNet)

## 📊 Results

본 논문은 뇌, 유방, 심장, 폐, 신장, 간, 폴립, 전립선, 망막, 피부 등 다양한 장기 및 병변 분할 작업에 대한 실험 결과들을 정리하였다.

### 주요 관찰 결과
-   **Non-Transformer의 경향:** 폴립, 전립선, 피부 병변 분할과 같이 경계가 모호한 작업에서는 에지 정보가 중요하므로 **Spatial Attention**이 Channel Attention보다 더 많이 사용되는 경향이 있다.
-   **Transformer의 경향:** 
    -   2D 데이터셋에서는 Hybrid Encoder-CNN Decoder 구조가 우수한 성능을 보이는 경우가 많다.
    -   3D 데이터셋(예: BraTS 뇌종양 분할)에서는 Pure Transformer Encoder-Decoder 구조가 더 좋은 성능을 내는 경향이 있다.
-   **정량적 지표:** 분석된 대부분의 논문에서 Dice Score, Jaccard Index, Hausdorff Distance (HD) 등을 주요 지표로 사용하고 있으며, Attention의 도입이 Baseline 대비 유의미한 성능 향상을 가져왔음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 Attention 기법이 단순한 성능 향상을 넘어, 모델이 입력 데이터의 어느 부분에 집중하고 있는지를 보여줌으로써 '블랙박스'인 신경망에 대한 **해석 가능성(Interpretability)**을 제공한다는 점을 강조한다.

### 한계 및 향후 과제
1.  **Task-specific Attention의 부족:** 현재 대부분의 Attention 모듈은 범용적으로 설계되어 있다. 하지만 폐결절(다양한 크기)이나 뇌종양(종양과 정상 조직의 구분)처럼 작업 특성에 맞게 설계된 전용 Attention 메커니즘 연구가 필요하다.
2.  **강건성(Robustness) 문제:** Attention 맵이 때때로 잘못된 영역에 집중하여 예측 오류를 일으키는 경우가 있으나, 이에 대한 실패 사례 분석(Failure analysis) 연구가 부족하다.
3.  **표준 평가의 부재:** 데이터셋, 전처리 방식, 데이터 분할 방법이 논문마다 달라 객관적인 비교가 어렵다. 표준화된 벤치마크와 평가 프로세스 구축이 시급하다.
4.  **복잡도 문제:** Transformer의 높은 계산 및 메모리 비용은 의료 현장 적용의 걸림돌이다. 효율적인 Self-attention 계산법과 데이터 희소성 문제를 해결하기 위한 자가 지도 학습(Self-supervised learning)의 결합이 필요하다.

## 📌 TL;DR

이 논문은 의료 영상 분할 분야에서 사용되는 **Non-Transformer 기반 Attention**과 **Transformer 기반 Attention**을 "무엇을(What), 어떻게(How), 어디에(Where)" 사용했는지를 기준으로 체계적으로 분석한 종합 서베이 보고서이다. CNN의 지역적 특징 추출 능력과 Transformer의 전역적 문맥 파악 능력을 결합하는 추세가 뚜렷하며, 향후에는 단순한 범용 모듈을 넘어 **특정 의료 작업의 특성을 반영한 맞춤형 Attention 설계**와 **계산 효율성 최적화**가 핵심 연구 방향이 될 것으로 전망한다.