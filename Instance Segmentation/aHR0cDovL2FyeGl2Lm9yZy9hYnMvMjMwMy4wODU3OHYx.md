# SIM: Semantic-aware Instance Mask Generation for Box-Supervised Instance Segmentation

Ruihuang Li, Chenhang He, Yabin Zhang, Shuai Li, Liyi Chen, Lei Zhang (2023)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Bounding Box 주석(annotation)만을 사용하여 인스턴스 분할(Instance Segmentation)을 수행하는 **Box-supervised Instance Segmentation (BSIS)**이다. 픽셀 단위의 정밀한 마스크 주석은 생성 비용이 매우 높기 때문에, 상대적으로 비용이 저렴한 박스 주석만으로 고성능의 분할 모델을 학습시키는 것이 중요하다.

기존의 BSIS 방법들은 주로 인접 픽셀 간의 색상이나 텍스처와 같은 **low-level image features (low-level 이미지 특징)**에 의존하여 유사도를 계산하고 이를 통해 pseudo mask를 생성한다. 그러나 이러한 방식은 전경 객체가 배경이나 주변의 다른 객체와 색상이 비슷할 경우 객체 간 경계를 명확히 구분하지 못하고 서로 뭉쳐버리는(blending) 문제가 발생한다. 즉, 객체의 전역적인 구조나 고수준의 의미론적 정보(high-level semantic information)를 활용하지 못한다는 점이 핵심적인 한계이다.

따라서 본 연구의 목표는 객체의 의미론적 정보를 명시적으로 활용하는 **Semantic-aware Instance Mask (SIM)** 생성 패러다임을 개발하여, 저수준 특징의 한계를 극복하고 보다 정확한 인스턴스 마스크를 생성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 저수준의 픽셀 유사도에만 의존하지 않고, 데이터셋 수준의 **Prototypes (프로토타입)**를 구축하여 객체의 고수준 의미론적 특성을 학습하는 것이다.

1.  **Semantic-aware Instance Mask Generation**: 클래스별 특징 중심점(feature centroids)인 프로토타입을 구축하여, 픽셀이 어떤 클래스에 속하는지를 결정하는 의미론적 가이드를 제공한다. 이를 통해 배경과 객체가 유사한 색상을 가졌더라도 의미론적 차이를 통해 전경을 더 정확히 식별할 수 있다.
2.  **Self-Correction Mechanism**: 프로토타입 기반의 의미론적 마스크는 동일 클래스의 서로 다른 인스턴스를 구분하지 못하는 한계가 있다. 이를 해결하기 위해 인스턴스별 가중치 기반의 통합 전략을 사용하여 잘못 활성화된 영역을 수정하고 개별 인스턴스를 구분하는 **instance-aware**한 마스크로 정교화한다.
3.  **Online Weakly-supervised Copy-Paste**: 약지도 학습 환경에서 부족한 정밀 마스크 데이터를 보완하기 위해, 생성된 pseudo mask를 활용하여 객체를 복제하고 붙여넣는 Copy-Paste 증강 기법을 제안한다. 특히 중요도 샘플링(importance sampling)을 통해 고품질 마스크를 가진 객체를 우선적으로 활용하여 폐색(occlusion) 상황과 희귀 클래스에 대한 학습 능력을 높인다.

## 📎 Related Works

인스턴스 분할 연구는 Mask R-CNN과 같은 Fully-supervised 방식에서 시작하여, 최근에는 비용 절감을 위해 Box-level 또는 Image-level 주석만을 사용하는 **Weakly-supervised Instance Segmentation (WSIS)**으로 확장되었다.

기존의 BSIS 접근 방식은 크게 두 가지로 나뉜다. 첫째는 GrabCut이나 MCG와 같은 전통적인 제안 기법을 사용하여 offline으로 pseudo mask를 생성하는 방식인데, 이는 파이프라인이 복잡하고 반복적인 단계가 많아 번거롭다. 둘째는 BBTP나 BoxInst와 같이 픽셀 간의 pairwise affinity(쌍별 유사도)를 손실 함수에 도입하여 end-to-end 학습을 수행하는 방식이다. 

하지만 BoxInst와 같은 최신 방법들은 인접한 픽셀들이 비슷한 색상을 가지면 같은 라벨을 가질 것이라고 단순 가정한다. 이는 객체 내부의 전역적 구조를 무시하는 방식이며, 전경과 배경의 색상이 유사한 경우 심각한 성능 저하를 초래한다. 본 논문은 이러한 저수준 특징 의존성을 탈피하고, 고수준의 semantic prototype을 도입함으로써 기존 방법론들과 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인
SIM 프레임워크는 기본 분할 네트워크 $F_{seg}$와 모멘텀 업데이트를 통해 안정성을 유지하는 $F'_{seg}$로 구성된다. 전체 프로세스는 **프로토타입 생성 $\rightarrow$ 의미론적 맵 생성 $\rightarrow$ 자기 수정(Self-Correction) $\rightarrow$ 최종 Pseudo Mask 도출 $\rightarrow$ 네트워크 학습** 순으로 진행된다.

### 1. Semantic-aware Instance Mask Generation
객체의 고유한 구조적 정보를 캡처하기 위해 클래스 $c$당 $L$개의 서브 센터(sub-centers)인 프로토타입 $P^c = \{p^c_1, \dots, p^c_L\}$을 유지한다.

- **Pseudo Semantic Map 생성**: 입력 이미지의 특징 맵 $Z$의 각 픽셀 벡터 $z_i$와 프로토타입 간의 코사인 유사도를 계산하여 확률 맵 $M^c_S$를 생성한다.
  $$M^c_{S,i} = \sigma\left(\max_{l=1}^L \{ \langle z_i, p^c_l \rangle / \tau \}\right)$$
  여기서 $\sigma$는 sigmoid 함수이며, $\tau$는 표현의 집중도를 조절하는 파라미터이다.

- **Prototype Update**: 프로토타입은 Sinkhorn-Knopp 알고리즘을 이용한 최적 운송(Optimal Transport) 문제로 정의하여 픽셀-클러스터 할당 $Q$를 최적화하고, 이를 통해 계산된 중심점을 모멘텀 평균 방식으로 업데이트한다.
  $$p^c_{l|t} = \gamma \cdot p^c_{l|t-1} + (1-\gamma) \cdot p^c_{n,l}$$

### 2. Self-Correction (자기 수정)
의미론적 맵 $M_S$는 클래스 구분은 잘 하지만, 동일 클래스의 서로 다른 인스턴스를 구분하지 못한다. 이를 위해 **Positive mask weighting** 전략을 사용한다.

- **인스턴스 확률 맵 $M_I$ 생성**: 앵커 프리 검출기(예: FCOS)에서 생성된 여러 긍정 샘플(positive samples) 중, GT 박스와의 IoU가 높은 샘플에 더 큰 가중치를 부여하여 통합한다.
  $$w_{pos} = e^{\mu \cdot IoU}$$
  이 가중치를 통해 정교한 인스턴스 중심의 마스크 $M_I$를 얻는다.

- **최종 Pseudo Mask $\hat{M}$ 결정**: 의미론적 맵 $M_S$와 인스턴스 맵 $M_I$를 선형 결합하여 최종 확률 맵을 생성하고, 두 개의 임계값 $\tau_{high}, \tau_{low}$를 적용하여 확신이 높은 영역만 pseudo label로 사용한다.
  $$\hat{M}_{k,i}^{prob} = (1-\alpha) \cdot M^k_{S,i} + \alpha \cdot M^k_{I,i}$$

### 3. Online Weakly-supervised Copy-Paste
학습 데이터의 효율성을 높이기 위해 memory bank를 구축하여 과거의 이미지와 pseudo mask를 저장한다. 
- **중요도 샘플링**: 마스크의 확신도(score) $S$를 계산하여, 고품질 마스크를 가진 객체를 우선적으로 선택해 다른 이미지에 붙여넣는다.
- 이를 통해 복잡한 폐색 상황이나 희귀 카테고리에 대한 학습 데이터를 인위적으로 늘려 모델의 강건성을 높인다.

### 4. 손실 함수 (Objective Function)
최종 학습은 다음의 세 가지 손실 함수의 합으로 이루어진다.
$$L_{seg} = L_{lowlevel} + \lambda_1 L_{pseudo} + \lambda_2 L_{paste}$$
- $L_{lowlevel}$: 기존 BoxInst에서 사용된 저수준 픽셀 유사도 손실.
- $L_{pseudo}$: SIM을 통해 생성된 pseudo mask와 예측값 사이의 BCE 및 Dice loss.
- $L_{paste}$: Copy-Paste로 생성된 데이터에 대한 손실.

## 📊 Results

### 실험 설정
- **데이터셋**: COCO (train2017, val2017, test-dev2017) 및 PASCAL VOC 2012.
- **베이스라인**: CondInst, Mask2Former.
- **백본**: ResNet-101-FPN, ResNet-DCN-101-BiFPN, Swin-B-FPN.

### 주요 결과
- **정량적 성능**: COCO 데이터셋에서 ResNet-101-FPN 백본 기준, SIM은 **35.3% AP**를 달성하여 BoxInst(33.2% AP)와 BoxLevelSet(33.4% AP)을 유의미하게 앞섰다. 특히 Swin-B-FPN 백본 사용 시 **40.2% AP**라는 높은 성능을 보였다.
- **객체 크기별 성능**: 작은 객체($AP_S$)에서 특히 큰 향상을 보였는데, 이는 proposed Copy-Paste 기법이 작은 객체에 대한 학습 데이터를 효과적으로 증강했기 때문으로 분석된다.
- **정성적 결과**: 시각화 결과, 기존 방법론들이 배경과 객체를 구분하지 못하거나 동일 클래스의 여러 객체를 하나로 뭉뚱그려 예측하는 반면, SIM은 객체의 전체적인 형태를 잘 유지하며 개별 인스턴스를 명확히 분리해내는 능력을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **상호보완적 감독**: 본 논문은 저수준의 local affinity($L_{lowlevel}$)와 고수준의 global semantic($L_{pseudo}$) 정보를 동시에 사용함으로써, 국소적인 경계 정보와 전역적인 객체 구조 정보를 모두 확보하였다.
- **프로토타입의 효과**: 데이터셋 전체에서 추상화된 프로토타입을 사용함으로써 특정 이미지에 국한된 노이즈를 필터링하고, 클래스 고유의 내재적 속성을 학습할 수 있었다.
- **자기 수정의 필요성**: $\alpha$ 값에 대한 ablation study 결과, $M_S$만 사용하거나 $M_I$만 사용했을 때보다 두 맵을 통합했을 때 성능이 가장 좋았다. 이는 의미론적 일관성과 인스턴스 구분 능력이 서로 보완 관계에 있음을 입증한다.

### 한계 및 논의사항
- **하이퍼파라미터 의존성**: $\mu, \tau, \alpha$ 등 다수의 하이퍼파라미터가 도입되었으며, 이에 따른 성능 변동이 존재한다.
- **계산 비용**: 프로토타입 업데이트를 위한 Sinkhorn-Knopp 알고리즘과 모멘텀 인코더의 도입으로 인해 학습 과정의 복잡도가 다소 증가했을 가능성이 있다.

## 📌 TL;DR

본 논문은 Bounding Box만으로 인스턴스 분할을 수행하는 BSIS에서, 저수준 특징 의존성을 극복하기 위해 **클래스별 프로토타입 기반의 의미론적 가이드(SIM)**를 제안한다. 특히 **Self-Correction** 모듈을 통해 의미론적 마스크를 인스턴스 단위로 정교화하고, **Online Copy-Paste**를 통해 데이터 부족 문제를 해결함으로써 COCO 및 PASCAL VOC 데이터셋에서 SOTA 성능을 달성하였다. 이 연구는 약지도 학습 환경에서 고수준의 의미론적 정보를 어떻게 효율적으로 pseudo label 생성에 활용할 수 있는지에 대한 중요한 방향성을 제시한다.