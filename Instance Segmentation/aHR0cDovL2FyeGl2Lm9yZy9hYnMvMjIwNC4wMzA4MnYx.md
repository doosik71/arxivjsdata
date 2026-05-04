# Instance Segmentation of Unlabeled Modalities via Cyclic Segmentation GAN

Leander Lauenburg, Zudi Lin, Ruihan Zhang, Márcia dos Santos, Siyu Huang, Ignacio Arganda-Carreras, Edward S. Boyden, Hanspeter Pfister, and Donglai Wei (2022)

## 🧩 Problem to Solve

본 논문은 레이블이 없는 새로운 이미징 모달리티(Imaging Modality)에서 객체 분할(Instance Segmentation)을 수행하는 문제를 다룬다. 특히 3D 신경 세포핵(neuronal nuclei) 분할과 같이 전문가의 정밀한 주석(annotation)이 필요한 작업의 경우, 새로운 데이터셋을 수집할 때마다 대량의 레이블을 생성하는 것은 비용과 시간이 매우 많이 소요되는 작업이다.

기존의 접근 방식은 크게 두 가지이다. 하나는 다양한 데이터로 사전 학습된 범용 모델(Generalist model)을 직접 적용하는 것이고, 다른 하나는 CycleGAN과 같은 도메인 변환(Domain Translation) 모델을 이용해 이미지를 변환한 뒤 별도의 분할 모델을 적용하는 순차적(Sequential) 방식이다. 그러나 전자는 레이블이 없는 새로운 도메인에 적응하기 어렵고, 후자는 변환 모델이 하위 작업인 분할(segmentation)의 목적을 고려하지 않고 학습되어 오류가 전파될 수 있으며 시스템 복잡도가 증가한다는 한계가 있다.

따라서 본 논문의 목표는 이미지 변환과 인스턴스 분할을 하나의 통합된 프레임워크 내에서 동시에 수행하여, 레이블이 없는 타겟 도메인의 이미지를 효과적으로 분할하는 CySGAN을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 이미지 변환과 분할을 독립된 단계로 나누지 않고, 하나의 생성기(Generator)가 변환된 이미지와 분할 표현(Segmentation representation)을 동시에 출력하도록 설계한 것이다.

가장 중심적인 기여는 다음과 같다.

1. **통합 프레임워크(Unified Framework):** 이미지 변환과 분할을 동시에 수행하는 Cyclic Segmentation GAN(CySGAN)을 제안하여 파이프라인의 복잡성을 줄이고 두 작업 간의 상호 최적화를 가능하게 하였다.
2. **준지도 학습 손실 함수 도입:** 레이블이 없는 타겟 도메인의 데이터를 효율적으로 활용하기 위해 구조적 일관성 손실(Structural Consistency Loss)과 분할 기반 적대적 손실(Segmentation-based Adversarial Loss)을 도입하였다.
3. **데이터 증강 및 복구 전략:** 데이터 증강으로 인해 손상된 이미지를 깨끗한 이미지로 복구하도록 사이클 일관성(Cycle Consistency)을 강제함으로써 모델의 정규화 성능을 높였다.
4. **NucExM 데이터셋 구축:** 확장 현미경(Expansion Microscopy, ExM) 이미지에 대해 정밀하게 주석이 달린 새로운 3D 세포핵 데이터셋을 구축하여 공개하였다.

## 📎 Related Works

### 관련 연구 및 한계

- **Unpaired Image-to-Image Translation:** CycleGAN과 같은 모델은 짝지어지지 않은 이미지 쌍을 이용해 도메인 간 변환을 수행한다. 하지만 이는 주로 시각적 유사성에 치중하며, 이후 수행될 분할 작업의 정확도를 보장하지 않는다.
- **3D Instance Segmentation:** Cellpose나 StarDist와 같은 모델은 3D 현미경 이미지 분할에서 뛰어난 성능을 보이지만, 기본적으로 지도 학습(Supervised Learning)에 의존하므로 새로운 모달리티에 적용하려면 많은 양의 레이블이 필요하다.
- **Combination of Translation and Segmentation:** CyCADA나 SUSAN과 같은 연구가 변환과 분할의 결합을 시도하였다. 특히 SUSAN은 변환과 분할을 동시에 수행하는 개념을 제시하였으나, 2D 시맨틱 분할(Semantic Segmentation)에 국한되어 있으며 타겟 도메인에 대한 준지도 학습(Semi-supervised) 관점의 손실 함수가 부족하다는 한계가 있다.

### 차별점

CySGAN은 SUSAN과 달리 더 어려운 과제인 **3D 인스턴스 분할**을 다루며, 타겟 도메인의 레이블 부재를 해결하기 위해 구조적 일관성 및 적대적 손실을 추가하여 준지도 학습 능력을 강화하였다.

## 🛠️ Methodology

### 전체 시스템 구조

CySGAN은 두 개의 생성기 $F$와 $B$를 사용하며, 각 생성기는 입력 이미지를 받아 **변환된 이미지**와 **분할 표현(Segmentation representations)**을 동시에 출력한다.

- $F: I_X \rightarrow (I_Y, S_X)$ : 소스 도메인 $X$를 타겟 도메인 $Y$로 변환하고 분할 결과 $S_X$를 예측한다.
- $B: I_Y \rightarrow (I_X, S_Y)$ : 타겟 도메인 $Y$를 소스 도메인 $X$로 변환하고 분할 결과 $S_Y$를 예측한다.

### 상세 구성 요소 및 손실 함수

전체 손실 함수 $\mathcal{L}$은 크게 세 가지 부분의 합으로 정의된다:
$$\mathcal{L} = \mathcal{L}_{\text{translation}} + \mathcal{L}_{\text{supervised\_seg}} + \mathcal{L}_{\text{semi-supervised\_seg}}$$

#### 1. 이미지 변환 손실 ($\mathcal{L}_{\text{translation}}$)

CycleGAN의 구조를 따라 적대적 손실($\mathcal{L}_{GAN}$)과 사이클 일관성 손실($\mathcal{L}_{cyc}$)을 사용한다.

- **적대적 손실:** 변환된 이미지가 타겟 도메인의 실제 이미지 분포와 구별되지 않도록 학습한다.
- **사이클 일관성 손실:** $x \rightarrow F(x) \rightarrow B(F(x)) \approx x$ 가 되도록 하여 구조적 보존을 강제한다.
$$\mathcal{L}_{cyc}(F, B) = \|B(\hat{y}_i)_{[I]} - x_i\|_1 + \|F(\hat{x}_i)_{[I]} - y_i\|_1$$

#### 2. 지도 학습 분할 손실 ($\mathcal{L}_{\text{supervised\_seg}}$)

레이블이 있는 소스 도메인 $X$에 대해 적용된다. 본 논문은 U3D-BCD 방식을 따라 세 가지 표현(Foreground mask $B$, Contour map $C$, Distance transform $D$)을 예측한다.

- $B, C$ 채널: Binary Cross-Entropy (BCE) 손실 사용.
- $D$ 채널: Mean Squared Error (MSE) 손실 사용.
$$\mathcal{L}_{seg}(F) = \mathcal{L}_{bce}(F(x_i)_{B}^{[S]}, x_{sB}) + \mathcal{L}_{bce}(F(x_i)_{C}^{[S]}, x_{sC}) + \|F(x_i)_{D}^{[S]} - x_{sD}\|_2^2$$

#### 3. 준지도 학습 분할 손실 ($\mathcal{L}_{\text{semi-supervised\_seg}}$)

레이블이 없는 타겟 도메인 $Y$의 성능을 높이기 위해 도입되었다.

- **구조적 일관성 손실 ($\mathcal{L}_{sc}$):** 서로 다른 생성기가 예측한 분할 결과가 동일한 구조를 가져야 한다는 점을 이용한다.
$$\mathcal{L}_{sc}(F, B) = \|B(y_i)_{[S]} - F(B(y_i)_{[I]})_{[S]}\|_1$$
- **분할 기반 적대적 손실:** 타겟 도메인에서 예측된 분할 표현들의 분포가 소스 도메인의 정답(Ground-truth) 분할 표현 분포와 유사해지도록 판별기 $D_{S_X}$를 통해 학습한다.

### 학습 절차 및 특이사항

- **3D U-Net:** $F$와 $B$의 아키텍처로 사용된다.
- **데이터 증강 전략:** 무작위 결손, 블러링 등의 증강을 적용하되, 사이클 일관성 손실을 계산할 때는 증강된 이미지가 아닌 '깨끗한(clean)' 이미지와 비교하도록 하여 모델이 손상된 영역을 복구하는 능력을 갖추게 하였다.
- **분리 학습:** 분할 손실을 계산할 때 변환된 이미지를 detach하여, 분할 목표가 이미지 변환 결과(시각적 품질)를 저해하지 않도록 설계하였다.

## 📊 Results

### 실험 설정

- **데이터셋:** 소스 도메인으로는 NucMM-Z(전자 현미경, EM)를, 타겟 도메인으로는 새롭게 구축한 NucExM(확장 현미경, ExM)을 사용하였다.
- **측정 지표:** Average Precision (AP-50)을 사용하여 인스턴스 분할 성능을 평가하였다.
- **비교 대상:**
  - 사전 학습 모델: Cellpose, StarDist
  - 순차적 파이프라인: Histogram matching + Segm, CycleGAN + Segm (소스$\rightarrow$타겟 및 타겟$\rightarrow$소스 방향 모두 테스트)

### 주요 결과

정량적 결과는 Table 2에 제시되어 있으며, CySGAN이 모든 기준 모델보다 우수한 성능을 보였다.

- **정량적 성과:** CySGAN은 평균 AP-50 $0.931$을 기록하며, 가장 강력한 베이스라인인 'CycleGAN + Segm ($I_X \rightarrow I_Y$)' 모델($0.874$)보다 절대적으로 **5.7% 더 높은 성능**을 보였다.
- **정성적 분석:** Cellpose는 거짓 음성(False Negative)이 많이 발생하였고, StarDist는 객체 경계가 불분명하고 겹치는 현상이 발생하였다. 반면 CySGAN은 객체의 경계를 훨씬 정확하게 포착하였다.
- **Ablation Study:** 준지도 학습 손실(Semi-supervised loss)을 제거하면 성능이 $0.927 \rightarrow 0.878$로 하락하며, 데이터 증강을 제거하면 $0.761$까지 급격히 하락하여 두 요소 모두 필수적임을 확인하였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 이미지 변환과 분할이라는 두 가지 서로 다른 과제를 하나의 네트워크로 통합함으로써, 변환 모델이 분할 작업에 최적화된 특징을 학습하도록 유도하였다. 특히 타겟 도메인에 레이블이 전혀 없는 상황에서 구조적 일관성과 적대적 학습을 통해 준지도 학습 효과를 낸 점이 매우 효과적이었다. 또한, 데이터 증강 과정에서 발생한 노이즈를 복구하는 방향으로 사이클 일관성을 정의하여 모델의 강건성(Robustness)을 높인 점이 돋보인다.

### 한계 및 논의사항

논문에서 제시된 방법론은 소스 도메인과 타겟 도메인 간의 **인스턴스 구조(instance structures)가 기본적으로 유사하다**는 가정에 기반하고 있다. 만약 두 도메인 간의 객체 모양이나 분포가 근본적으로 다르다면, 단순한 이미지 변환과 구조적 일관성만으로는 한계가 있을 것이다. 향후 연구에서는 이러한 구조적 차이가 큰 도메인 간의 전이 학습을 어떻게 해결할 것인지가 중요한 과제가 될 것이다.

## 📌 TL;DR

**요약:** 레이블이 없는 새로운 3D 현미경 이미지 모달리티의 세포핵 분할을 위해, 이미지 변환과 인스턴스 분할을 동시에 수행하는 통합 GAN 프레임워크인 **CySGAN**을 제안하였다. 준지도 학습 손실과 정교한 데이터 증강 전략을 통해, 레이블이 없는 데이터에서도 기존의 순차적 변환-분할 방식이나 범용 모델보다 훨씬 높은 정확도를 달성하였다.

**의의:** 전문가의 주석 비용이 매우 높은 바이오메디컬 이미징 분야에서, 기존의 레이블된 데이터를 활용해 새로운 촬영 기법(모달리티)의 데이터를 효율적으로 분석할 수 있는 실용적인 방법론을 제시하였다.
