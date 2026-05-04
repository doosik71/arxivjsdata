# Camouflaged Instance Segmentation In-The-Wild: Dataset, Method, and Benchmark Suite

Trung-Nghia Le, Yubo Cao, Tan-Cong Nguyen, Minh-Quan Le, Khanh-Duy Nguyen, Thanh-Toan Do, Minh-Triet Tran, and Tam V. Nguyen (2021)

## 🧩 Problem to Solve

본 논문은 이미지 내에서 배경과 유사한 외형을 가져 식별이 어려운 **Camouflaged Instance Segmentation (CIS)**이라는 새로운 과제를 정의하고 해결하고자 한다. 

기존의 위장 객체 탐지 연구들은 주로 **Camouflaged Object Segmentation (COS)**에 집중해 왔으며, 이는 픽셀 단위로 위장 여부만을 판별하는 Region-level의 세그멘테이션에 그쳤다. 따라서 한 장면 내에 여러 개의 위장 객체가 존재할 때, 각각의 개별 인스턴스를 분리하여 식별하는 Instance-level의 분석은 불가능했다.

또한, 기존 연구들은 "이미지 내에 항상 위장 객체가 존재한다"는 가정을 전제로 수행되었다. 하지만 실제 환경(In-the-wild)에서는 위장 객체가 없을 수도 있으며, 주변 맥락에 따라 위장 여부가 결정된다. 따라서 본 논문은 다음과 같은 목표를 가진다.
1. 위장 객체를 개별 인스턴스 단위로 분리하는 **Camouflaged Instance Segmentation** 태스크의 정의.
2. 실제 환경을 모사하여 위장 객체가 존재하지 않는 이미지까지 포함하는 대규모 데이터셋 구축.
3. 다양한 최신 인스턴스 세그멘테이션 모델을 평가할 수 있는 벤치마크 수립 및 성능 향상을 위한 융합 프레임워크 제안.

## ✨ Key Contributions

본 논문의 핵심 기여는 위장 인스턴스 세그멘테이션을 위한 데이터셋 제공, 벤치마크 분석, 그리고 모델 융합 방법론 제안으로 요약된다.

1. **CAMO++ 데이터셋 구축**: 기존 CAMO 데이터셋을 확장하여 5,500장의 이미지(위장 이미지 2,700장, 비위장 이미지 2,800장)와 93개의 카테고리를 포함하는 대규모 데이터셋을 구축하였다. 특히 계층적 픽셀 단위 어노테이션(메타 카테고리, 세부 카테고리, 바운딩 박스, 인스턴스 마스크)을 제공한다.
2. **벤치마크 수이트(Benchmark Suite) 제공**: 최신 인스턴스 세그멘테이션 방법론들을 다양한 시나리오(위장 객체 존재 가정 여부 등)에서 평가하여 해당 태스크의 난이도와 현재 기술 수준을 분석하였다.
3. **Camouflage Fusion Learning (CFL) 프레임워크 제안**: 단일 모델이 모든 상황에서 최적의 성능을 낼 수 없다는 점에서 착안하여, 이미지의 컨텍스트를 학습해 최적의 인스턴스 세그멘테이션 모델을 선택하고 융합하는 프레임워크를 제안하였다.

## 📎 Related Works

### 1. Camouflaged Object Segmentation (COS)
초기 연구들은 색상, 강도, 모양 등 handcrafted low-level feature를 사용하였으나 단순한 배경에서만 효과적이었다. 최근에는 ANet, SINet, MirrorNet, TINet과 같은 딥러닝 기반 모델들이 제안되었지만, 이들은 모두 픽셀을 위장/비위장 레이블로 매핑하는 수준이며, 객체의 개수나 개별 인스턴스의 정체성을 구분하지 못하는 한계가 있다.

### 2. Camouflage Datasets
기존의 CamouflagedAnimals, CHAMELEON, CAMO, COD, MoCA 데이터셋 등이 존재한다. 그러나 COD 데이터셋의 경우 위장 이미지에 대해서만 어노테이션을 제공하며, 실제 환경처럼 위장 객체가 없는 경우를 포함하지 않는다. 또한, 대부분의 데이터셋이 인스턴스 수준의 마스크를 충분히 제공하지 않아 CIS 태스크를 수행하기에 부적합했다.

### 3. Instance Segmentation
인스턴스 세그멘테이션은 크게 Proposal-based 접근법(Two-stage 및 Single-stage)과 Segmentation-based 접근법으로 나뉜다. 본 논문은 현재 SOTA 성능을 보이는 Proposal-based 방법론(예: Mask R-CNN, YOLACT 등)에 집중하여 벤치마크를 수행하였다.

## 🛠️ Methodology

### 1. CAMO++ 데이터셋 구성
- **데이터 수집**: 인터넷에서 "camouflaged", "hidden" 등의 키워드와 동물/사람/환경 키워드를 조합해 수집하였다.
- **구성**: 위장 이미지 2,700장과 비위장 이미지(LVIS 데이터셋에서 선별) 2,800장으로 구성되어 약 50:50의 비율을 유지한다.
- **계층 구조**: 생물학적 분류(Biology taxonomic structure)와 시각적 특징 기반 분류(Vision taxonomic structure)라는 두 가지 계층적 구조를 제공하여 분석의 편의성을 높였다.

### 2. Camouflage Fusion Learning (CFL) 프레임워크
CFL은 여러 개의 인스턴스 세그멘테이션(IS) 모델 중 특정 이미지에 가장 적합한 모델을 예측하여 그 결과를 선택하는 구조이다.

**전체 파이프라인:**
1. **개별 모델 학습**: Mask R-CNN, Cascade Mask R-CNN, MS R-CNN, RetinaMask, CenterMask 등 5개의 SOTA 모델을 CAMO++ 데이터셋으로 독립적으로 학습시킨다.
2. **최적 모델 탐색 (Model Search)**: Greedy 알고리즘(Algorithm 1)을 사용하여 각 이미지에 대해 가장 높은 Average Precision (AP)을 기록하는 모델을 찾아 pseudo-label을 생성한다.
3. **모델 예측기 학습**: Vision Transformer (ViT-Base16)를 사용하여 입력 이미지 $x$가 주어졌을 때 어떤 IS 모델이 최적일지 예측하는 모델 예측기 $f$를 학습시킨다.

**손실 함수 (Loss Function):**
전체 손실 함수 $L$은 다음과 같이 세그멘테이션 손실과 모델 예측 손실의 합으로 정의된다.
$$L = L_{segm} + L_{pred}$$

여기서 세그멘테이션 손실 $L_{segm}$은 탐색 알고리즘을 통해 선택된 최적 모델 $c$에 대해서만 가중치를 두어 계산된다.
$$L_{segm}(x) = \sum_{g=1}^{M} c_g(x) \times L_{g}^{ins}(g(x), y)$$
($M$은 모델의 수, $c_g$는 최적 모델 여부를 나타내는 지표(0 또는 1), $L_{g}^{ins}$는 각 모델의 고유 세그멘테이션 손실이다.)

모델 예측 손실 $L_{pred}$는 다항 로지스틱 회귀를 위한 Cross Entropy Loss를 사용한다.
$$L_{pred}(x) = -c(x) \cdot \log(f(x))$$

## 📊 Results

### 1. 실험 설정 및 지표
- **지표**: COCO 스타일의 $AP$, $AP_{50}$, $AP_{75}$ 및 스케일별 $AP$ ($AP_S, AP_M, AP_L$), 그리고 Average Recall ($AR$)을 사용하였다.
- **백본**: ResNet50-FPN, ResNet101-FPN, ResNeXt101-FPN을 사용하였다.
- **시나리오**: 
    - **Setting 1 (In-the-wild)**: 위장 객체가 없을 수도 있는 모든 이미지를 대상으로 테스트.
    - **Setting 2 (Assumption)**: 모든 이미지에 위장 객체가 존재한다고 가정하고 위장 이미지로만 테스트.

### 2. 정량적 결과
- **Setting 1**: CFL 프레임워크가 모든 지표에서 SOTA 성능을 달성하였다. 특히 ResNeXt101-FPN 백본 사용 시 $AP$ 25.1을 기록하며 다른 단일 모델들을 크게 상회하였다.
- **Setting 2**: 위장 객체 존재 가정 시 성능이 전반적으로 상승하였으며, CFL은 ResNeXt101-FPN에서 $AP$ 42.8이라는 최고 성능을 보였다.
- **추가 학습 효과**: 비위장 이미지(Non-camouflage images)를 학습에 추가했을 때, 위장 인스턴스 세그멘테이션 성능이 향상됨을 확인하였다. 특히 CFL(ResNeXt101)의 경우 $AP$가 42.8까지 상승하였다.

### 3. 데이터셋 일반화 평가 (Generalization)
CAMO++와 COD 데이터셋 간의 교차 학습-테스트를 수행한 결과, CAMO++에서 학습한 모델이 COD 테스트셋에서 더 높은 성능을 보였으며, 이는 CAMO++ 데이터셋이 더 다양하고 도전적인 샘플을 포함하고 있음을 시사한다.

## 🧠 Insights & Discussion

### 1. 강점 및 성과
본 연구는 단순한 객체 탐지를 넘어 인스턴스 단위의 위장 세그멘테이션이라는 새로운 과제를 정의하고, 이를 위한 고품질의 대규모 데이터셋을 구축하였다는 점에서 가치가 크다. 또한, 단일 모델의 한계를 극복하기 위해 컨텍스트 기반의 모델 융합(CFL) 방식을 도입하여 성능을 유의미하게 끌어올렸다.

### 2. 한계 및 실패 사례 분석
실험 결과, 다음과 같은 경우에 모델이 여전히 취약함을 발견하였다.
- **Tiny Instances**: 크기가 극도로 작은 객체는 위치 파악 및 세그멘테이션이 매우 어렵다.
- **Extreme Resemblance**: 배경과 색상 및 텍스처가 거의 완벽하게 일치하는 경우 인간조차 구분이 어려우며, 모델 역시 실패하는 경향이 있다.
- **Occlusion & Overlap**: 객체 간의 겹침이나 폐쇄가 발생했을 때 인스턴스를 잘못 분리하거나 오분류하는 사례가 발생하였다.

### 3. 비판적 해석
SOTA 모델들을 융합했음에도 불구하고 Setting 1에서의 Top-1 $AP$가 25 이하라는 점은, 실제 야생 환경에서의 위장 인스턴스 세그멘테이션이 여전히 매우 어려운 문제임을 보여준다. 단순한 모델 융합보다는 위장 객체 특유의 텍스처 분석이나 주변 컨텍스트를 더 깊게 활용하는 새로운 아키텍처 설계가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 배경과 구분이 어려운 위장 객체를 개별 인스턴스로 분리하는 **Camouflaged Instance Segmentation (CIS)** 태스크를 최초로 제안하고, 이를 위한 대규모 데이터셋인 **CAMO++**를 구축하였다. 또한, 여러 SOTA 모델 중 최적의 모델을 이미지별로 선택해 결합하는 **Camouflage Fusion Learning (CFL)** 프레임워크를 통해 기존 단일 모델보다 향상된 성능을 달성하였다. 이 연구는 위장 객체 탐지 분야를 Region-level에서 Instance-level로 확장시켰으며, 향후 자율 주행, 야생 동물 보존, 의료 영상 분석 등의 분야에 기여할 가능성이 높다.