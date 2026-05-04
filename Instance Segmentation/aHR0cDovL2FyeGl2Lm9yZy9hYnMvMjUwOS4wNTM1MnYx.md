# Unsupervised Instance Segmentation with Superpixels

Cuong Manh Hoang (2025)

## 🧩 Problem to Solve

인스턴스 분할(Instance Segmentation)은 로보틱스, 자율 주행 등 다양한 컴퓨터 비전 응용 분야에서 핵심적인 역할을 수행한다. 현재의 주류 모델들은 대규모의 인간 주석(human annotations) 데이터를 통해 높은 성능을 달성하고 있으나, 이러한 데이터를 수집하는 데에는 막대한 비용과 시간이 소요된다는 치명적인 단점이 있다.

이를 해결하기 위해 비지도 학습(Unsupervised learning) 기반의 인스턴스 분할 연구들이 진행되어 왔다. 하지만 기존의 비지도 학습 방식들은 몇 가지 한계를 가진다. 예를 들어 TokenCut과 같은 방식은 이미지당 단 하나의 객체만 분할할 수 있으며, CutLER의 MaskCut 방식은 이미지당 분할할 객체의 수를 하이퍼파라미터로 미리 정해두어야 하므로 과소 분할(under-segmentation) 또는 과잉 분할(over-segmentation) 문제가 발생한다. 또한, 많은 모델이 성능 향상을 위해 다단계의 self-training 과정을 거치는데, 이는 학습 시간을 과도하게 증가시킨다.

따라서 본 논문의 목표는 인간의 주석 없이도 효율적이고 효과적으로 클래스 불가지론적(class-agnostic) 인스턴스 마스크를 생성할 수 있는 새로운 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 자기지도학습(self-supervised) 기반의 고수준 특징(high-level features)과 슈퍼픽셀(superpixels) 및 색상 정보와 같은 저수준 특징(low-level features)을 결합하여, 정밀한 인간 주석 없이도 네트워크를 효과적으로 학습시키는 것이다.

주요 기여 사항은 다음과 같다.

1. **슈퍼픽셀 가이드 마스크 손실 함수($L_{sgm}$) 제안**: 슈퍼픽셀, 거친 마스크(coarse mask), 색상 정보를 통합하여 학습의 효과를 높이는 새로운 손실 함수를 도입하였다.
2. **적응형 손실 함수($L_{ad}$) 기반의 Self-training**: 예측된 마스크의 전역적 안정성(holistic stability)을 측정하여 신뢰도에 따라 손실 가중치를 조절하는 적응형 손실 함수를 통해 효율적으로 마스크 품질을 개선하였다.
3. **범용적 적용 가능성 증명**: COCO, PASCAL VOC, KITTI뿐만 아니라 특수 도메인인 SAR(Synthetic Aperture Radar) 이미지 데이터셋에서도 우수한 성능을 보임을 입증하였다.

## 📎 Related Works

기존의 인스턴스 분할 연구는 크게 지도 학습과 비지도 학습으로 나뉜다. Mask R-CNN과 같은 Top-down 방식이나 Mask2Former와 같은 Transformer 기반의 최신 모델들은 압도적인 성능을 보이지만, 앞서 언급한 대로 방대한 양의 정답 레이블이 필수적이다.

비지도 학습 분야에서는 ViT(Vision Transformer)의 자기지도 특징을 활용하는 시도들이 있었다. LOST와 TokenCut은 단일 객체 분할에 국한되었으며, FreeSOLO와 unMORE는 도메인 내의 레이블 없는 데이터로 사전 학습을 진행해야 한다는 제약이 있다. CutLER는 MaskCut을 통해 다중 객체 분할을 가능하게 했으나, 객체 수 지정 문제와 긴 self-training 시간이라는 한계가 존재한다.

본 논문은 이러한 한계를 극복하기 위해 객체 수를 미리 정하지 않는 MultiCut 알고리즘을 도입하고, 슈퍼픽셀을 통해 마스크의 경계 정밀도를 높임으로써 기존 비지도 학습 방식과의 차별점을 둔다.

## 🛠️ Methodology

본 프레임워크의 전체 파이프라인은 거친 마스크 생성, 슈퍼픽셀 가이드 학습, 그리고 적응형 self-training의 세 단계로 구성된다.

### 1. 거친 마스크 생성 (Coarse Masks Generation)

먼저, 사전 학습된 self-supervised ViT를 사용하여 이미지에서 고수준 특징 $F \in \mathbb{R}^{N \times N \times E}$를 추출한다. 이후 RAMA(Rapid bottom-up multicut algorithm)를 적용하여 이미지 내의 모든 잠재적 객체 마스크를 생성한다.

생성된 마스크 중 품질이 낮은 것을 걸러내기 위해 마스크 필터(mask filter)를 사용한다. 패치 간의 코사인 유사도를 통해 생성된 어피니티 맵(affinity map) $A$를 이용하여, 마스크 내부의 결합력과 경계의 분리력을 평가하는 지표 $R(M)$을 계산한다.
$$R(M) = \frac{1}{|M_{inner}|} \sum_{i \in M_{inner}} A_i - \frac{1}{|M_{edge}|} \sum_{j \in M_{edge}} A_j$$
이 값이 높은 상위 $Q\%$의 마스크만을 선택하여 학습에 사용한다.

### 2. 슈퍼픽셀 가이드 마스크 손실 함수 (Superpixel-guided Mask Loss)

분할 네트워크로는 SOLO를 채택하였으며, 저수준 특징인 슈퍼픽셀(MCG 알고리즘 사용)을 통해 마스크의 정밀도를 보완한다. 전체 손실 함수 $L_{sgm}$은 다음과 같이 하드 손실($L_{hard}$)과 소프트 손실($L_{soft}$)의 합으로 정의된다.
$$L_{sgm} = L_{hard} + L_{soft}$$

- **Hard Loss ($L_{hard}$)**: 슈퍼픽셀 $S_k$의 평균 색상 $\mu_k$와 픽셀 색상 $C_i$의 유사도 $\delta_{k,i}$를 계산하여, 슈퍼픽셀이 전경(foreground)일 확률 $P_k$를 구한다. 거친 마스크를 통해 슈퍼픽셀 전체가 전경이거나 배경인 경우에만 하드 레이블 $y_k \in \{0, 1\}$을 부여하고 교차 엔트로피 손실을 계산한다.
- **Soft Loss ($L_{soft}$)**: 국소적인 정보만 사용하는 하드 손실의 한계를 극복하기 위해, 슈퍼픽셀 간의 색상 유사도를 기반으로 MST(Minimum Spanning Tree) 그래프를 구축한다. 전역적 쌍방향 어피니티 $\psi_{k,l}$를 통해 소프트 레이블 $\hat{P}_k$를 생성하며, 예측값 $P_k$와의 $L_1$ 거리를 통해 학습한다.
$$\psi_{k,l} = \exp\left(-\frac{\max_{\forall(m,n) \in E_{k,l}} w_{m,n}}{\alpha_2}\right)$$

### 3. 적응형 손실을 통한 Self-training (Adaptive Loss)

학습된 모델의 예측 결과로 다시 모델을 학습시키는 self-training 단계에서는 마스크의 신뢰도를 평가하는 전역적 안정성(holistic stability) $Z_i$를 도입한다. 이는 여러 체크포인트에서 생성된 마스크들 간의 IoU 합으로 계산된다.
$$Z_i = \sum_{j=1}^{e-1} \text{IoU}(m_{e,i}, m_{j,i})$$
정규화된 점수 $\bar{Z}_i$와 픽셀이 경계로부터 떨어진 거리 $\phi_j$를 가중치로 사용하여 적응형 손실 함수 $L_{ad}$를 계산한다. 이는 신뢰도가 낮은 마스크나 노이즈가 많은 경계 지역의 영향을 줄여 효율적으로 품질을 높인다.

## 📊 Results

### 실험 설정

- **데이터셋**: ImageNet으로 사전 학습 후, COCO val2017, COCO 20K, PASCAL VOC, UVO, KITTI, SSDD 등에서 zero-shot 평가를 수행하였다.
- **지표**: 클래스 불가지론적(class-agnostic) AP (Average Precision)를 사용하였다.
- **구현**: ViT-B/8 (DINO) 및 ResNet-101 기반의 SOLO 모델을 사용하였다.

### 주요 결과

1. **정량적 성능**: COCO 20K 데이터셋에서 $AP_{box}$ 18.8%, $AP_{mask}$ 14.8%를 기록하며 기존 SOTA 모델인 unMORE를 상회하는 성능을 보였다. PASCAL VOC에서는 $AP$ 기준 33.1%를 달성하여 기존 방법들보다 유의미하게 높은 성능을 보였다.
2. **범용성**: 실제 주행 환경인 KITTI 데이터셋과 특수 영상인 SAR 이미지(SSDD 데이터셋)에서도 가장 높은 성능을 기록하여, 다양한 도메인에 대한 일반화 능력을 입증하였다.
3. **소거 연구(Ablation Study)**: 마스크 필터, $L_{hard}$, $L_{soft}$, $L_{ad}$를 단계적으로 적용했을 때 성능이 지속적으로 향상됨을 확인하였다. 특히 슈퍼픽셀을 사용하지 않았을 때보다 사용했을 때 $AP_{box}$가 4.6% 증가하여 슈퍼픽셀의 중요성을 확인하였다.
4. **범용 분할(Universal Segmentation) 확장**: U2Seg 프레임워크 내에서 CutLER를 본 제안 방법으로 대체했을 때, 인스턴스 분할뿐만 아니라 시맨틱 분할(mIoU 4.3% 상승) 및 파놉틱 분할(PQ 4.5% 상승) 성능이 모두 향상되었다.

## 🧠 Insights & Discussion

본 논문은 비지도 학습 기반 인스턴스 분할에서 고질적인 문제였던 '거친 마스크의 부정확성'을 저수준 특징인 슈퍼픽셀과 전역적 색상 어피니티를 통해 효과적으로 해결하였다. 특히, 단순히 데이터만 늘리는 것이 아니라 전역적 안정성(holistic stability)이라는 지표를 통해 self-training의 신뢰도를 높인 점이 인상적이다.

**강점**:

- 슈퍼픽셀을 활용해 마스크 경계의 정밀도를 높여 인간 주석에 가까운 세밀한 분할이 가능하다.
- 객체 수를 미리 정할 필요가 없는 MultiCut 알고리즘을 통해 유연한 객체 탐지가 가능하다.
- 단 한 번의 self-training 라운드만으로도 다회 반복 학습을 수행한 기존 모델보다 높은 성능을 낸다.

**한계 및 논의**:

- 본 모델은 클래스 불가지론적(class-agnostic) 분할에 집중하고 있으며, 각 객체가 어떤 카테고리에 속하는지에 대한 분류(Classification) 기능은 포함하고 있지 않다.
- self-supervised ViT의 특징 추출 능력에 크게 의존하고 있으므로, ViT가 학습되지 않은 매우 특수한 도메인에서는 성능 저하가 발생할 가능성이 있다.

## 📌 TL;DR

본 논문은 인간의 주석 없이 인스턴스 분할을 수행하기 위해 **자기지도 특징(ViT)과 슈퍼픽셀(Superpixel)을 결합한 새로운 프레임워크**를 제안한다. 특히 슈퍼픽셀의 색상 정보를 이용한 **전역-국소 통합 손실 함수($L_{sgm}$)**와 예측 마스크의 안정성을 기반으로 한 **적응형 손실 함수($L_{ad}$)**를 통해 비지도 학습의 한계를 극복하고 SOTA 성능을 달성하였다. 이 연구는 레이블링 비용이 높은 의료 영상이나 특수 위성 영상 분할 분야에 즉시 적용될 가능성이 매우 높다.
