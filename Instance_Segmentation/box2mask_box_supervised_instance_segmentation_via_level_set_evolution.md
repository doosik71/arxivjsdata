# Box2Mask: Box-supervised Instance Segmentation via Level-set Evolution

Wentong Li, Wenyu Liu, Jianke Zhu, Miaomiao Cui, Risheng Yu, Xiansheng Hua, Lei Zhang (2022)

## 🧩 Problem to Solve

인스턴스 분할(Instance Segmentation)은 이미지 내의 객체별로 픽셀 단위의 마스크(mask)를 생성하는 작업으로, 자율 주행이나 의료 영상 분석 등 다양한 분야에서 필수적이다. 하지만 기존의 완전 지도 학습(Fully Supervised) 방식은 모든 객체에 대해 정밀한 픽셀 단위 마스크 라벨을 요구하며, 이는 막대한 비용과 시간이 소요되는 매우 고된 작업이다.

이를 해결하기 위해 단순한 바운딩 박스(Bounding Box) 주석만을 사용하는 Box-supervised Instance Segmentation 연구가 진행되어 왔다. 기존의 박스 기반 방식들은 주로 인접한 픽셀 간의 유사도(Pairwise Affinity)를 모델링하는 방식을 사용했다. 그러나 이러한 접근법은 "비슷한 색상이나 픽셀은 같은 라벨을 가질 것"이라는 지나치게 단순한 가정에 기반하고 있어, 배경이 복잡하거나 객체 간의 외관이 유사한 경우 성능이 크게 저하되는 한계가 있다. 따라서 본 논문의 목표는 바운딩 박스라는 약한 감독(Weak Supervision) 하에서도 정밀한 객체 경계를 찾아낼 수 있는 강건한 인스턴스 분할 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 고전적인 변분법적 접근 방식인 **Level-set Evolution 모델을 딥러닝 네트워크에 통합**하여, 픽셀 단위 라벨 없이도 바운딩 박스만으로 정확한 마스크를 예측하는 것이다.

주요 기여 사항은 다음과 같다.

1. **Box2Mask 프레임워크 제안**: Level-set 모델의 에너지 함수를 최소화하는 과정을 딥러닝 학습 과정에 포함시켜, 반복적으로 경계선을 최적화하는 단일 단계(Single-shot) 인스턴스 분할 방식을 제안하였다.
2. **다양한 아키텍처 지원**: CNN 기반 및 Transformer 기반의 두 가지 프레임워크를 개발하여 Level-set 진화를 구현하였다.
3. **강건한 경계 진화 전략**: 저수준 이미지 특징(Low-level image features)과 고수준 딥러닝 특징(High-level deep features)을 동시에 활용하고, 박스 투영 함수(Box projection function)를 통해 초기 Level-set을 자동으로 설정하여 진화의 안정성을 높였다.
4. **Local Consistency Module (LCM) 도입**: 픽셀 유사도 및 공간적 관계를 고려한 어피니티 커널(Affinity kernel)을 통해 지역적 일관성을 확보함으로써, 영역 기반 Level-set의 고질적인 문제인 강도 불균일성(Intensity inhomogeneity) 문제를 완화하였다.

## 📎 Related Works

### 1. Fully-supervised Instance Segmentation

Mask R-CNN과 같은 ROI 기반 방식이나 SOLO와 같은 ROI-free 방식, 그리고 최근의 Transformer 기반(Mask2Former 등) 방식들이 존재한다. 이들은 매우 정밀한 결과를 내놓지만, 픽셀 단위 마스크 라벨에 대한 의존도가 너무 높아 새로운 도메인에 적용하기 어렵다는 한계가 있다.

### 2. Box-supervised Instance Segmentation

바운딩 박스만을 사용하여 마스크를 예측하려는 시도로, BBTP, BoxInst, DiscoBox 등이 있다. 이들은 주로 픽셀 간의 Pairwise Affinity를 모델링하여 end-to-end 학습을 수행한다. 하지만 앞서 언급했듯이, 단순한 유사도 기반 모델링은 복잡한 배경이나 유사한 객체가 인접한 상황에서 노이즈에 취약하다는 한계가 있다.

### 3. Level-set based Image Segmentation

Level-set 방법은 에너지 함수를 통해 암시적으로 곡선을 표현하고 이를 최소화하여 경계를 찾는 방식이다. 기존의 딥러닝 기반 Level-set 연구(Levelset R-CNN 등)는 대부분 픽셀 단위의 정답 마스크를 사용하여 학습하는 완전 지도 학습 방식이었다. 본 논문은 이러한 Level-set의 강력한 경계 최적화 능력을 **박스 주석만으로 학습 가능하게 만들었다**는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

Box2Mask는 **Backbone $\rightarrow$ Instance-aware Decoder (IAD) $\rightarrow$ Box-level Matching $\rightarrow$ Level-set Evolution** 순으로 구성된다.

### 1. Instance-aware Decoder (IAD)

각 인스턴스의 고유한 특성을 임베딩하여 전체 이미지 크기의 마스크 맵을 동적으로 생성한다.

- **CNN-based IAD**: SOLOv2와 유사하게 동적 컨볼루션(Dynamic Convolution)을 사용한다. 커널 학습 네트워크가 인스턴스별 고유 커널 $K_{i,j}$를 생성하고, 이를 통합 마스크 특징 $F_{mask}$에 적용하여 $M_{i,j} = K_{i,j} * F_{mask}$를 계산한다.
- **Transformer-based IAD**: MaskFormer에서 영감을 얻어 Transformer 디코더를 통해 $N$개의 인스턴스 쿼리를 학습한다. MSDeformAtten 레이어를 통해 고해상도 마스크 특징 $F_{mask}$를 추출하고, 학습된 쿼리 커널 $K$와 도트 프로덕트(Dot product)를 수행하여 $M = K \cdot F_{mask}$를 생성한다.

### 2. Box-level Matching Assignment

박스 주석만을 사용하여 어떤 예측 마스크가 긍정 샘플(Positive sample)인지 결정한다.

- **CNN 기반**: 박스의 중심 영역에 예측 위치 $(i,j)$가 포함되면 긍정 샘플로 지정한다.
- **Transformer 기반**: 헝가리안 알고리즘(Hungarian algorithm)을 사용한다. 매칭 비용 $C$는 다음과 같이 정의된다:
  $$C = \beta_1 C_{inst} + \beta_2 C_{cate}$$
  여기서 $C_{inst}$는 예측 마스크와 정답 박스를 각각 x축, y축으로 투영하여 1D Dice 계수로 계산한 공간적 차이이며, $C_{cate}$는 클래스 분류 오차(Cross Entropy)이다.

### 3. Level-set Evolution

본 논문의 핵심으로, 바운딩 박스 내부에서 마스크 경계를 반복적으로 최적화한다.

**에너지 함수**: Chan-Vese 모델을 기반으로 하며, 다음과 같은 에너지 함수 $F$를 최소화한다.
$$F(\phi, I, c_1, c_2, B) = \int_{\Omega \in B} |I^*(x,y) - c_1|^2 \sigma(\phi(x,y)) dxdy + \int_{\Omega \in B} |I^*(x,y) - c_2|^2 (1 - \sigma(\phi(x,y))) dxdy + \gamma \int_{\Omega \in B} |\nabla \sigma(\phi(x,y))| dxdy$$

- $\phi(x,y)$는 Level-set 함수이며, $\sigma$는 Heaviside 함수 대신 미분 가능한 Sigmoid 함수를 사용하여 수렴 속도를 높였다.
- $c_1, c_2$는 각각 경계 내부와 외부의 평균 강도이다.
- $\gamma$는 경계선의 길이를 정규화하는 가중치이다.

**입력 데이터 및 초기화**:

- **데이터 통합**: 단순 이미지 $I^u$뿐만 아니라, Tree Filter를 통해 구조적 특징을 보존한 고수준 딥러닝 특징 $I^f$를 함께 사용하여 에너지 함수를 구성한다.
- **초기화**: 박스 투영 함수를 통해 예측 마스크와 정답 박스 간의 투영 차이를 계산함으로써, 최적화 시작 단계에서 대략적인 초기 경계 $\phi_0$를 자동으로 설정한다.

**Local Consistency Module (LCM)**:
지역적 강도 불균일성 문제를 해결하기 위해 픽셀 강도 어피니티 $A^p_{i,j}$와 공간적 어피니티 $A^s_{i,j}$를 결합한 커널을 정의하고, 이를 통해 예측된 $\phi$를 반복적으로 정제하여 지역적 일관성을 확보한다.

### 4. 학습 및 추론

- **손실 함수**: $L = w L_{cate} + L_{inst}$ 형태로 구성된다. 여기서 $L_{inst}$는 위에서 정의한 미분 가능한 Level-set 에너지 함수 그 자체이다.
- **추론(Inference)**: Level-set 진화 과정은 학습 시에만 사용되며, 추론 시에는 네트워크가 직접 마스크를 출력하므로 반복적인 최적화 과정 없이 효율적으로 동작한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Pascal VOC, COCO, iSAID(원격 탐사), LiTS(의료), ICDAR2019 ReCTS(텍스트) 등 5개 벤치마크를 사용하였다.
- **백본**: ResNet-50, ResNet-101, Swin-Transformer (B/L)를 사용하였다.

### 주요 결과

1. **일반 장면 (Pascal VOC, COCO)**:
   - ResNet-101 백본 기준, Box2Mask-T는 Pascal VOC에서 43.2% AP를 기록하여 기존 SOTA인 BoxInst(36.5%)를 크게 상회하였다.
   - COCO 데이터셋에서 Swin-L 백본을 사용한 Box2Mask-T는 **42.4% mask AP**를 달성하였으며, 이는 완전 지도 학습 기반의 방법들과 대등한 수준이다.
2. **특수 도메인**:
   - **iSAID (원격 탐사)**: 복잡한 배경 속에서도 Box2Mask-C(ResNet-101)가 26.6% AP를 기록하며, 완전 지도 학습 방식인 SOLO(23.5%)보다 높은 성능을 보였다.
   - **LiTS (의료)**: 의료 영상의 특징인 전경과 배경의 유사성 속에서도 고수준 특징을 활용한 Level-set 진화가 효과적으로 작동하여 55.3% AP를 달성하였다.
   - **ReCTS (텍스트)**: 텍스트 박스만으로도 정교한 문자 마스크를 생성할 수 있음을 보였으며, 이를 통해 얻은 박스 AP가 53.9%에 달했다.

### 분석 및 성능 지표

- **추론 속도**: Box2Mask-T (ResNet-101)는 약 7.9 FPS의 속도로 동작하며, 기존 Box-supervised 방법들보다 약간 느리지만 훨씬 높은 정확도를 제공한다.
- **Ablation Study**:
  - 박스 투영 초기화, 저수준/고수준 특징의 결합, LCM 모듈의 도입이 모두 성능 향상에 기여함을 확인하였다.
  - 특히 Level-set 진화를 이미지 전체가 아닌 바운딩 박스 영역($\Omega \in B$)으로 제한했을 때 성능이 크게 향상되었다.

## 🧠 Insights & Discussion

### 강점

본 연구는 고전적인 Level-set 모델을 현대적인 딥러닝 프레임워크에 성공적으로 이식하였다. 특히, 단순히 픽셀 간의 유사도를 보는 것이 아니라 **에너지 함수 최소화라는 변분법적 원리를 학습 목표로 설정**함으로써, 박스 주석만으로도 완전 지도 학습에 근접하는 정밀한 경계를 찾아낼 수 있음을 입증하였다. 또한, 다양한 도메인(의료, 항공, 텍스트)에서의 실험을 통해 제안 방법의 범용성을 증명하였다.

### 한계 및 논의사항

- **작은 객체에 대한 취약성**: 실험 결과에서 작은 객체($AP_S$)에 대한 성능이 상대적으로 낮게 나타났다. 이는 바운딩 박스 내부에서 작은 객체가 가지는 특징량이 부족하여 배경과 구분하기 어렵기 때문으로 분석된다.
- **계산 비용**: Level-set의 수식적 복잡성으로 인해 추론 속도가 기존의 단순한 Pairwise Affinity 기반 모델들보다는 다소 느리다. 다만, 추론 시에는 반복 연산을 제외하므로 실용적인 수준의 속도를 유지하고 있다.

### 비판적 해석

본 논문은 "박스-마스크 간의 갭"을 좁히는 데 집중하였으며, 특히 Transformer 기반의 IAD와 Level-set의 결합이 매우 강력한 시너지를 낸다는 것을 보여주었다. 다만, Level-set 초기화 단계에서 여전히 정답 박스(GT Box)에 의존하고 있으므로, 실제 환경에서 검출기(Detector)가 예측한 부정확한 박스를 사용할 때 성능이 얼마나 하락할지에 대한 분석이 추가되었다면 더 완벽했을 것이다.

## 📌 TL;DR

**Box2Mask**는 고전적인 **Level-set Evolution 모델과 딥러닝을 결합**하여, 픽셀 단위 라벨 없이 **바운딩 박스만으로 정밀한 인스턴스 분할**을 수행하는 프레임워크이다. Chan-Vese 에너지 함수를 최소화하는 과정을 학습에 도입하고, 저수준/고수준 특징과 지역 일관성 모듈(LCM)을 통해 경계를 최적화한다. COCO 데이터셋에서 **42.4% AP**를 기록하며 박스 기반 학습의 한계를 넘어 완전 지도 학습 수준의 성능에 도달하였으며, 이는 향후 라벨링 비용을 획기적으로 줄이면서도 고성능의 분할 모델을 구축하는 데 중요한 역할을 할 것으로 기대된다.
