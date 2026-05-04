# Zero-Shot Instance Segmentation

Ye Zheng, Jiahong Wu, Yongqiang Qin, Faen Zhang, Li Cui (2021)

## 🧩 Problem to Solve

본 논문은 **Zero-Shot Instance Segmentation (ZSI)**라는 새로운 태스크를 정의하고 이를 해결하기 위한 방법론을 제시한다. 일반적인 인스턴스 세그멘테이션(Instance Segmentation)은 대규모의 레이블링된 데이터를 필요로 하지만, 의료나 제조 분야와 같이 전문 지식이 필요하거나 데이터 수집이 어려운 도메인에서는 모든 클래스에 대한 학습 데이터를 확보하는 것이 매우 어렵다.

따라서 본 연구의 목표는 학습 단계에서 보지 못한 클래스, 즉 **unseen classes**에 대해서도 인스턴스 세그멘테이션을 수행할 수 있는 모델을 구축하는 것이다. 구체적으로는 학습 시에는 seen classes의 데이터만을 사용하고, 테스트 시에는 seen 및 unseen 클래스 모두에 대해 정밀한 픽셀 수준의 마스크를 생성하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **시각적 특징과 시맨틱 개념 간의 매핑 관계(Visual-Semantic Mapping Relationship)**를 학습하여 이를 unseen 클래스로 전이하는 것이다. 이를 위해 다음과 같은 설계 전략을 제안한다.

1. **Zero-Shot Detector 및 Semantic Mask Head (SMH)**: 사전 학습된 단어 벡터(word-vectors)를 활용하여 시각적 특징을 시맨틱 공간으로 투영하고, 이를 통해 unseen 클래스를 탐지하고 세그멘테이션한다.
2. **Background Aware RPN (BA-RPN)**: 단순한 단어 벡터 기반의 배경 표현의 한계를 극복하기 위해, 이미지로부터 직접 배경의 시맨틱 표현을 학습하는 RPN을 제안한다.
3. **Synchronized Background Strategy (Sync-bg)**: 학습 및 추론 과정에서 BA-RPN이 학습한 배경 벡터를 검출기와 SMH에 동기화하여, 이미지마다 변화하는 배경에 유연하게 대응하는 동적 적응형(dynamic adaptive) 배경 표현을 구현한다.
4. **ZSI 벤치마크 구축**: MS-COCO 데이터셋을 기반으로 ZSI를 평가하기 위한 실험 프로토콜과 벤치마크를 최초로 제시하였다.

## 📎 Related Works

기존의 인스턴스 세그멘테이션 방법론(Mask R-CNN, FCIS 등)은 강력한 지도 학습(supervised learning)에 의존하므로, 학습 샘플이 없는 새로운 카테고리로 확장하는 것이 불가능하다.

최근의 Zero-shot Learning 연구들은 주로 Zero-shot Classification에 집중해 왔으나, 이는 단일 객체만을 다루므로 복잡한 실제 장면을 처리하기에 부적합하다. 이를 해결하기 위해 Zero-shot Object Detection (ZSD)과 Zero-shot Semantic Segmentation (ZSS)이 제안되었다. 하지만 ZSD는 바운딩 박스 수준의 결과만을 제공하고, ZSS는 이미지 전체에 대한 시맨틱 맵을 생성할 뿐 개별 인스턴스를 구분하여 세그멘테이션하지는 못한다.

본 논문은 이러한 한계를 극복하여, 개별 인스턴스에 대한 픽셀 수준의 세그멘테이션을 수행하는 ZSI를 제안함으로써 ZSD와 ZSS의 간극을 메운다. 또한, 기존 ZSD 연구들이 배경 클래스를 고정된 단어 벡터로 표현하여 성능이 저하되었던 점을 지적하며, 이를 동적으로 학습하는 방식을 통해 차별성을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
본 모델은 Backbone 네트워크를 통해 추출된 시각적 특징을 기반으로 **BA-RPN $\rightarrow$ Zero-shot Detector $\rightarrow$ Semantic Mask Head** 순으로 이어지는 파이프라인을 가진다.

### 1. Zero-Shot Detector
시각적 특징과 시맨틱 개념의 관계를 학습하기 위해 인코더-디코더 구조를 채택한다.
- **Encoder ($T_e$)**: RoI(Region of Interest)의 시각적 특징을 시맨틱 특징으로 인코딩한다.
- **Decoder ($T_d$)**: 인코딩된 시맨틱 특징을 다시 원래의 시각적 특징으로 복원한다.
- **Reconstruction Loss ($L_R$)**: 원래 특징 $O$와 복원된 특징 $R$ 사이의 평균 제곱 오차(MSE)를 최소화하여 더 변별력 있는 시각-시맨틱 정렬을 학습한다.
  $$L_R = \sum_{i=1}^{F} (O_i - R_i)^2$$
- **추론**: 테스트 시에는 디코더를 제거하고, 인코더가 생성한 시맨틱 특징과 seen/unseen 클래스의 단어 벡터($W_s, W_u$) 간의 행렬 곱을 통해 클래스 점수를 계산한다.

### 2. Semantic Mask Head (SMH)
인스턴스별 마스크를 생성하기 위해 시각-시맨틱 매핑을 픽셀 단위로 확장한 구조이다.
- **구조**: $1 \times 1$ 합성곱 층으로 구성된 인코더 $E$와 디코더 $D$를 사용한다. 인코더 $E$는 시각적 특징을 $300 \times 28 \times 28$ 크기의 시맨틱 특징 텐서로 변환한다. 여기서 각 채널은 단어 벡터의 차원을 의미한다.
- **Classification Module**: 인코더 이후에 고정된 $1 \times 1$ 합성곱 층($W_s\text{-Conv}, W_u\text{-Conv}$)을 배치한다. 이 층의 가중치는 각 클래스의 단어 벡터로 설정되며, 이를 통해 각 픽셀이 어떤 클래스에 속하는지 결정한다.
- **학습**: Detector와 마찬가지로 재구성 손실 $L_R$을 사용하여 시각-시맨틱 정렬의 품질을 높인다.

### 3. BA-RPN 및 Synchronized Background (Sync-bg)
배경 클래스가 고정된 단어 벡터로 표현될 때 발생하는 '배경-unseen 클래스 간의 혼동' 문제를 해결하기 위한 장치이다.
- **BA-RPN**: 이미지의 시각적 특징을 입력받아 배경-전경 이진 분류 점수를 계산하며, 이 과정에서 배경 클래스의 단어 벡터 $v_b$를 학습 가능한 파라미터로 둔다.
- **Sync-bg**: BA-RPN에서 학습된 $v_b$를 Zero-shot Detector와 SMH의 배경 벡터로 실시간 업데이트(동기화)한다. 이를 통해 모델은 이미지마다 다른 복잡한 배경에 적응적으로 대응할 수 있다.

### 4. 전체 손실 함수
전체 네트워크는 다음과 같은 결합 손실 함수를 통해 학습된다.
$$L_{ZSI} = L_{BA} + L_{ZSD} + L_{SMH}$$
- $L_{BA}$: 전경-배경 분류 cross-entropy loss 및 smooth $L_1$ regression loss.
- $L_{ZSD}$: 분류 loss, regression loss 및 재구성 손실 $\lambda_{ZSD} L_R$.
- $L_{SMH}$: 픽셀 단위 binary cross-entropy loss 및 재구성 손실 $\lambda_{SMH} L_R$.

## 📊 Results

### 실험 설정
- **데이터셋**: MS-COCO 2014 버전을 사용하여 48/17 split(seen 48, unseen 17)과 65/15 split(seen 65, unseen 15)으로 나누어 실험하였다.
- **평가 지표**: Recall@100 (IoU 0.4, 0.5, 0.6) 및 mAP (IoU 0.5)를 사용하였다.
- **비교 설정**: 일반적인 ZSI 설정과 seen/unseen을 동시에 예측해야 하는 Generalized ZSI (GZSI) 설정을 모두 평가하였다.

### 주요 결과
1. **ZSD 성능**: 제안 방법은 기존의 최신 ZSD 모델들(BLC, PL 등)보다 월등한 성능을 보였다. 48/17 split 기준 Recall@100에서 최대 36.99%의 향상을 기록하였다.
2. **ZSI 및 GZSI 성능**: baseline 대비 GZSI 설정에서 harmonic average (HM) 기준 Recall@100이 최대 11.72% 향상되었으며, mAP 또한 5.61% 증가하였다.
3. **구성 요소 분석 (Ablation Study)**: 
   - BA-RPN 단독 사용보다는 **BA-RPN과 Sync-bg를 함께 사용**했을 때 성능이 크게 향상되었다.
   - 특히 Sync-bg를 Detector와 SMH 전체에 적용했을 때 가장 높은 성능을 보였는데, 이는 검출 단계와 세그멘테이션 단계 사이의 배경 표현 일관성이 중요함을 시사한다.
4. **시맨틱 정보의 중요성**: Word2vec 대신 one-hot 벡터나 random baseline을 사용했을 때 성능이 급격히 저하됨을 확인하여, 사전 학습된 시맨틱 지식의 전이가 필수적임을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 Zero-shot Instance Segmentation이라는 도전적인 과제를 정의하고, 이를 위해 시각-시맨틱 매핑과 적응형 배경 학습이라는 두 가지 핵심 전략을 성공적으로 결합하였다.

**강점**:
- 단순한 분류/검출을 넘어 픽셀 수준의 세그멘테이션으로 Zero-shot Learning의 범위를 확장하였다.
- 배경 클래스를 고정된 상수로 취급하지 않고, 이미지 특성에 따라 동적으로 변하는 벡터로 처리함으로써 unseen 클래스의 오탐지율을 효과적으로 낮추었다.
- 인코더-디코더 구조를 통한 재구성 손실(reconstruction loss) 도입이 시각-시맨틱 정렬을 더욱 견고하게 만들었음을 실험적으로 증명하였다.

**한계 및 논의**:
- 본 연구는 word2vec과 같은 외부 시맨틱 벡터에 크게 의존하고 있다. 만약 시맨틱 공간 자체가 불완전하거나 시각적 특징과 괴리가 클 경우 성능 저하가 불가피할 수 있다.
- GZSI 설정에서 seen 클래스에 대한 성능 저하(seen-bias) 문제가 완전히 해결되었는지는 추가적인 분석이 필요해 보인다.
- 실험이 MS-COCO 데이터셋에 국한되어 있어, 논문에서 언급한 의료나 제조 분야의 실제 도메인 데이터에서도 동일한 효과가 나타날지는 검증되지 않았다.

## 📌 TL;DR

본 논문은 학습 시 보지 못한 클래스의 인스턴스를 세그멘테이션하는 **Zero-Shot Instance Segmentation (ZSI)** 태스크를 제안하고, 이를 위해 **시각-시맨틱 매핑 기반의 Detector 및 Mask Head**와 **동적 배경 학습 전략(BA-RPN & Sync-bg)**을 개발하였다. 실험 결과, 제안 방법은 기존 ZSD 모델들을 압도하는 성능을 보였으며, 특히 배경 표현의 동기화가 unseen 클래스 탐지에 핵심적인 역할을 함을 밝혔다. 이 연구는 데이터 부족 문제가 심각한 특수 도메인의 인스턴스 세그멘테이션 연구에 중요한 기초 baseline이 될 것으로 기대된다.