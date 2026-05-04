# MaskPlus: Improving Mask Generation for Instance Segmentation

Shichao Xu, Shuyue Lan, Qi Zhu (2019)

## 🧩 Problem to Solve

인스턴스 세그멘테이션(Instance Segmentation)은 객체 탐지(Object Detection)와 세그멘테이션(Segmentation)을 결합한 고난도 과제이다. Mask R-CNN과 같은 최신 접근 방식들은 대개 문제를 탐지 구성 요소와 마스크 생성 분기(Mask Generation Branch)의 두 부분으로 나누어 처리한다. 그러나 기존 연구들은 주로 탐지 성능을 높이는 데 집중하였으며, 정작 마스크 생성 분기에는 다음과 같은 심각한 한계점이 존재한다.

첫째, 마스크 분기가 RoI(Region of Interest) 내부의 특징만을 사용하므로 전역적인 세만틱 정보(Global Semantic Information)를 손실할 가능성이 크다. 둘째, 불완전한 바운딩 박스(Bounding Box)가 생성될 경우 마스크 분기의 전체적인 성능이 저하된다. 셋째, 단일 디컨볼루션(Deconvolutional) 레이어 중심의 단순한 구조와 경계 정교화(Boundary Refinement) 과정의 부재로 인해 결과물이 거칠게 생성된다. 마지막으로, 각 구성 요소의 학습 속도 차이로 인해 다중 작업 학습(Multitask Training) 과정에서 충돌이 발생하여 성능 저하가 일어날 수 있다.

본 논문의 목표는 이러한 마스크 생성 분기의 한계를 극복하기 위해 다섯 가지 새로운 기술을 제안하여 마스크 생성 능력을 향상시키고, 학습 과정에서 탐지 분기와 마스크 분기 간의 충돌을 줄이는 MaskPlus 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 마스크 생성 분기의 성능을 극대화하기 위해 독립적으로 적용 가능한 다섯 가지 기술을 도입하는 것이다.

1. **Contextual Fusion**: 전역 정보를 마스크 분기에 통합하여 RoI 외부의 문맥 정보를 활용한다.
2. **Deconvolutional Pyramid Module**: 다단계 세만틱 의미를 결합하여 더 정교한 마스크 예측을 수행한다.
3. **Improved Boundary Refinement**: Dense Connection 구조를 통해 객체의 경계선을 더 날카롭게 다듬는다.
4. **Quasi-multitask Learning**: 다양한 스케일의 레이블을 가이드로 사용하여 네트워크의 스케일 불변성(Scale Invariant Ability)을 높인다.
5. **Biased Training**: 마스크 분기의 초기 학습 속도를 조절하여 탐지 분기와의 학습 충돌을 방지한다.

## 📎 Related Works

기존의 인스턴스 세그멘테이션 연구는 크게 두 가지 방향으로 나뉜다. 하나는 유사한 콘텐츠를 그룹화하는 클러스터링(Clustering) 기반 방식이며, 다른 하나는 Faster R-CNN과 같은 객체 탐지 모델의 제안 영역(Proposal)을 활용하는 방식이다. 특히 Mask R-CNN은 RoIAlign을 통해 정렬 문제를 해결하며 표준적인 프레임워크로 자리 잡았다.

이후 PANet과 같은 연구들이 등장하여 백본 네트워크를 개선하고 경로 집계 네트워크(Path Aggregation Network)를 추가하는 등 성능 향상을 꾀했다. 하지만 저자들은 이러한 후속 연구들이 여전히 탐지 구성 요소의 개선에 치중해 있으며, 마스크 생성 분기 자체의 구조적 한계와 학습 전략의 문제는 충분히 다루지 않았음을 지적하며 MaskPlus의 차별성을 강조한다.

## 🛠️ Methodology

MaskPlus는 Mask R-CNN 프레임워크를 확장하며, 전체 파이프라인은 FPN(Feature Pyramid Network) 백본과 RoIAlign을 기반으로 한다. 구체적인 다섯 가지 기술의 상세 내용은 다음과 같다.

### 1. Contextual Fusion

RoI 내부 특징만으로는 전역적인 문맥(Context)을 파악하기 어렵고 RoI가 불완전할 경우 객체의 일부가 누락될 수 있다. 이를 해결하기 위해 RoIAlign 이전 단계의 FPN 마지막 레이어 특징에서 전체 이미지 크기의 제안(Full-image-size proposal)을 추출하는 새로운 분기를 생성한다. 이 분기는 3개의 합성곱 레이어(필터 수 512, 256, 256)를 거쳐 원래의 RoIAlign 특징에 더해진다. 이는 개별 RoI에 갇히지 않고 이미지 전체의 공간적 관계를 마스크 생성에 활용하게 한다.

### 2. Deconvolutional Pyramid Module

FPN의 구조에서 영감을 얻아, 먼저 업샘플링을 수행한 후 다시 다운샘플링을 수행하는 피라미드 모듈을 설계하였다. 구체적으로는 스트라이드 2인 디컨볼루션 레이어 세트와 그 뒤를 잇는 동일 수의 합성곱 레이어(스트라이드 2)로 구성된다. 이 모듈은 단순한 단일 업샘플링 대신 다단계의 세만틱 의미를 결합함으로써 다양한 크기의 객체에 대해 마스크 정확도를 높인다.

### 3. Improved Boundary Refinement

마스크의 경계가 흐릿해지는 문제를 해결하기 위해 경계 학습 전용 분기를 추가하였다. 기존 연구의 단순한 잔차 블록(Residual Block) 대신, Dense Connection을 갖춘 여러 개의 합성곱 모듈을 배치하여 학습 능력을 강화하였다. 각 모듈은 $\text{BatchNorm} \rightarrow \text{PReLu} \rightarrow \text{Conv(16 filters)} \rightarrow \text{BatchNorm} \rightarrow \text{PReLu} \rightarrow \text{Conv(4 filters)}$ 순으로 구성되며, 이후 모듈은 이전 모든 모듈의 특징을 연결(Concatenate)하여 입력으로 사용한다.

### 4. Quasi-multitask Learning

데이터 증강을 네트워크 끝단(출력부)에서 수행하는 전략이다. 학습 단계에서 원본 마스크 크기($28 \times 28$) 외에 $0.5\times(14 \times 14)$ 및 $2\times(56 \times 56)$ 크기의 마스크 분기를 병렬로 추가하여 손실 함수를 계산한다. 중요한 점은 이러한 다양한 스케일 분기는 학습 시 가이드 역할만 할 뿐, 추론 단계에서는 사용되지 않는다는 것이다. 이를 통해 추가적인 연산량 없이 스케일 불변성을 확보한다.

### 5. Biased Training

마스크 분기가 초기에 불안정한 피드백을 주어 탐지 분기의 학습을 방해하는 문제를 해결하기 위해 손실 함수에 가중치를 부여한다. 전체 다중 작업 손실 함수는 다음과 같이 정의된다.

$$L = L_{cls} + L_{box} + \alpha(L_{mask})$$

여기서 $L_{cls}$와 $L_{box}$는 각각 분류 및 바운딩 박스 손실이며, $L_{mask}$는 마스크 손실이다. 학습 초기에는 $\alpha$ 값을 1보다 크게(예: 1.5) 설정하여 마스크 분기가 빠르게 수렴하도록 유도함으로써 탐지 분기에 주는 부정적 영향을 줄인다. 이후 학습 후반부에는 $\alpha$를 1로 설정하여 정상적인 학습을 진행한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MS-COCO (훈련 115k, 검증 5k, 테스트 41k 이미지)
- **지표**: $\text{AP}, \text{AP}_{50}, \text{AP}_{75}$ 및 크기별 $\text{AP}_S, \text{AP}_M, \text{AP}_L$
- **구현**: Tensorpack 프레임워크 기반, ResNet-50-FPN(절제 연구) 및 ResNet-101-FPN(최종 모델) 사용.

### 주요 결과

- **Contextual Fusion**: 마스크 $\text{AP}$가 35.1에서 35.5로 상승하였으며, 특히 중대형 객체에서 효과가 컸다.
- **Deconvolutional Pyramid Module**: 마스크 $\text{AP}$가 35.4로 상승하였으며, 소형 및 중형 객체의 정확도가 개선되었다.
- **Improved Boundary Refinement**: 기존의 단순한 경계 정교화 방식보다 우수한 성능을 보였으며, 특히 중대형 객체의 정밀도가 향상되었다.
- **Quasi-multitask Learning**: $0.5\times$ 및 $2\times$ 스케일 학습 모두 성능 향상을 가져왔으며, 단일 스케일 감독의 오분류 문제를 방지했다.
- **Biased Training**: 탐지 성능($\text{AP}_{bb}$)과 마스크 성능 모두에서 이점이 확인되었다.

### 종합 성능 비교

최종 모델인 MaskPlus는 원본 Mask R-CNN보다 모든 지표에서 우수한 성능을 보였다. 특히 Cascade R-CNN 구조를 결합한 **MaskPlus+** 버전은 $\text{AP}$ 40.9를 달성하며 최신 기법들과 경쟁 가능한 수준의 성능을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 인스턴스 세그멘테이션의 성능 향상이 단순히 탐지기(Detector)의 정교함뿐만 아니라, 마스크 생성 분기의 구조적 설계와 학습 전략의 최적화에 달려 있음을 보여준다. 특히 Biased Training이라는 전략을 통해 다중 작업 학습에서 발생할 수 있는 상충 관계(Trade-off)를 해결하려 한 점이 인상적이다.

다만, 논문에서 언급되었듯이 ResNeXt-101, ASPP(Atrous Spatial Pyramid Pooling) 등 최신 백본 및 모듈을 사용하지 않았음에도 높은 성능을 냈다는 점은 본 연구의 제안 기법들이 범용적으로 적용 가능함을 시사한다. 하지만 이러한 최신 기술들을 추가로 통합했을 때 어느 정도의 성능 향상이 있을지에 대한 분석은 이루어지지 않았다. 또한, $\alpha$ 값의 설정이나 스케일 비율 등의 하이퍼파라미터가 실험적으로 결정된 부분이 많아, 다른 데이터셋에서의 일반화 성능에 대한 추가 검증이 필요해 보인다.

## 📌 TL;DR

본 논문은 Mask R-CNN의 마스크 생성 분기가 가진 한계를 극복하기 위해 **Contextual Fusion, Deconvolutional Pyramid, Boundary Refinement, Quasi-multitask Learning, Biased Training**이라는 다섯 가지 기술을 제안한 **MaskPlus**를 선보였다. 이 연구는 전역 문맥 활용과 정교한 경계 학습, 그리고 학습 전략의 최적화를 통해 인스턴스 세그멘테이션의 정확도를 유의미하게 향상시켰으며, 향후 다양한 객체 분할 아키텍처의 효율적인 마스크 생성 모듈 설계에 중요한 가이드라인을 제공할 것으로 기대된다.
