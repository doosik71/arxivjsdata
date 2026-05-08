# DRL-STNet: Unsupervised Domain Adaptation for Cross-modality Medical Image Segmentation via Disentangled Representation Learning

Hui Lin, Florian Schiffers, Santiago López-Tapia, Neda Tavakoli, Daniel Kim, Aggelos K. Katsaggelos (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation)에서 발생하는 **교차 모달리티(Cross-modality)** 데이터의 레이블 부족 문제를 해결하고자 한다. 예를 들어, CT 스캔 데이터에는 풍부한 레이블(Annotation)이 존재하지만, MRI 스캔 데이터에는 레이블이 없는 경우가 많다. 모든 모달리티에 대해 수동으로 레이블을 생성하는 것은 비용과 시간이 매우 많이 소모되며, 동일 환자에 대해 여러 모달리티의 데이터를 쌍(paired)으로 확보하는 것 또한 물류 및 일정상의 제약으로 인해 현실적으로 어렵다.

따라서 본 연구의 목표는 레이블이 있는 소스 도메인(Source domain, 예: CT)의 지식을 레이블이 없는 타겟 도메인(Target domain, 예: MRI)으로 전이시키는 **비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)** 프레임워크를 구축하여, 타겟 도메인에서의 분할 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **분리 표현 학습(Disentangled Representation Learning, DRL)**과 **자기 학습(Self-Training, ST)**을 결합하여 소스-타겟 간의 도메인 간극을 줄이는 것이다.

1. **DRL 기반 이미지 변환**: GAN 구조 내에서 이미지의 '콘텐츠(Content)'와 '스타일(Style)'을 분리하여 학습함으로써, 소스 이미지의 해부학적 구조(콘텐츠)를 유지한 채 타겟 모달리티의 특성(스타일)만을 입힌 합성 이미지를 생성한다. 이를 통해 쌍을 이루지 않는(unpaired) 데이터셋에서도 효과적인 이미지 변환이 가능하다.
2. **반복적 자기 학습(Iterative Self-Training)**: 합성된 타겟 이미지로 초기 모델을 학습시킨 후, 실제 타겟 이미지에 대해 생성한 의사 레이블(Pseudo-label)을 활용해 모델을 반복적으로 미세 조정(Fine-tuning)함으로써 일반화 성능을 극대화한다.

## 📎 Related Works

기존의 UDA 접근 방식 중 Jiang 등과 Yao 등은 복부 장기 분할을 위해 분리 표현 학습을 적용한 바 있다. 하지만 이러한 방법들이 다양한 소스와 시퀀스에 걸쳐 광범위한 복부 장기를 분할하는 데 얼마나 일반화될 수 있는지는 불분명했다. 최근 연구들은 변분 근사(Variational approximation)나 자기 학습 기법을 도입하여 견고성을 높이려 했으며, 일부에서는 GAN에 Transformer를 결합하여 슬라이스 방향의 연속성을 학습하려는 시도가 있었다.

DRL-STNet은 이러한 기존 연구들의 흐름을 이어받아, 이미지 수준의 변환뿐만 아니라 분할 모델의 학습 과정에 실제 타겟 데이터를 직접적으로 통합하는 자기 학습 루프를 설계함으로써 차별점을 둔다.

## 🛠️ Methodology

DRL-STNet 프레임워크는 크게 이미지 변환 단계와 분할 모델 학습 단계의 두 가지 주 공정으로 구성된다.

### 1. Image Translation (DRL-GAN)

이미지를 콘텐츠 표현 $c$와 스타일 표현 $s$로 분리하여 학습한다.

- **구조**: 공유 콘텐츠 인코더 $E_C$, 도메인별 스타일 인코더 $E_a^S, E_b^S$, 공유 디코더 $G$, 이미지 판별자 $D_a, D_b$, 콘텐츠 판별자 $D_c$로 구성된다.
- **작동 원리**: 소스 이미지 $x_a$에서 추출한 콘텐츠 $c_a$와 타겟 이미지 $x_b$에서 추출한 스타일 $s_b$를 결합하여 합성 이미지 $x_{ba} = G(c_a, s_b)$를 생성한다.

**손실 함수:**

- **재구성 손실(Reconstruction Loss)**: 인코더와 디코더가 원래 이미지를 정확히 복원하도록 강제한다.
$$L_{rec} = \mathbb{E}_{x_a \in \mathcal{X}_a} \|x_a - G(c_a, s_a)\| + \mathbb{E}_{x_b \in \mathcal{X}_b} \|x_b - G(c_b, s_b)\|$$
- **적대적 손실(Adversarial Loss)**: 이미지 수준과 콘텐츠 수준에서 도메인 간 정렬을 수행한다.
$$L_i^{adv} = \mathbb{E}_{x_i \in \mathcal{X}_i} [\log(D_i(x_i))] + \mathbb{E}_{c_j \in \mathcal{C}_j, s_i \in \mathcal{S}_i} [\log(1 - D_i(G(c_j, s_i)))]$$
($i, j \in \{a, b\}, i \neq j$)
- **콘텐츠 적대적 손실**: 콘텐츠 표현 $c$가 도메인에 독립적이도록 학습한다.
$$L_c^{adv} = \mathbb{E}_{c_b \in \mathcal{C}_b} [\log(D_c(c_b))] + \mathbb{E}_{c_a \in \mathcal{C}_a} [\log(1 - D_c(c_a))]$$

최종 최적화 목표는 다음과 같다:
$$\min_{(E_a^S, E_b^S, E_C, G)} \max_{(D_a, D_b, D_c)} L_{adv} + L_{rec}$$

### 2. Self-Training via Pseudo-Labeling

- **단계 3 (초기 학습)**: 소스 레이블 $Y_a$와 변환된 합성 이미지 $X_{ab}$ 쌍을 사용하여 3D nnU-Net 기반 분할 네트워크 $f$를 학습시킨다.
- **단계 4 (의사 레이블 생성)**: 학습된 모델 $f$를 레이블이 없는 실제 타겟 이미지 $X_b$에 적용하여 의사 레이블 $\hat{Y}_b = f(X_b)$를 생성한다.
- **단계 5 (미세 조정)**: 정교한 레이블을 가진 합성 이미지 $(X_{ab}, Y_a)$와 의사 레이블을 가진 실제 이미지 $(X_b, \hat{Y}_b)$를 함께 사용하여 모델을 미세 조정한다.
$$\mathcal{L} = \sum L_{seg}(Y_a, f(X_{ab})) + \sum L_{seg}(\hat{Y}_b, f(X_b))$$

## 📊 Results

### 실험 설정

- **데이터셋**: TCIA, LiTS, MSD 등 30개 이상의 의료 센터에서 수집된 CT 2,050건 및 MRI 4,000건 이상을 사용하였다. 테스트셋은 T1, T2, DWI 등 다양한 MRI 시퀀스를 포함한다.
- **평가 지표**: Dice Similarity Coefficient (DSC)와 Normalized Surface Dice (NSD)를 정확도 지표로 사용하였으며, 실행 시간과 GPU 메모리 사용량을 효율성 지표로 사용하였다.

### 주요 결과

- **정량적 성능**: FLARE 챌린지 데이터셋에서 DRL-STNet + ST(자기 학습) 조합은 평균 DSC 80.69%, NSD 80.69%를 기록하였다. 이는 UDA를 적용하지 않았을 때(DSC 6.13%)보다 비약적으로 상승한 것이며, 기존 SOTA 방법인 SIFA(DSC 67.52%)보다도 월등한 성능이다.
- **장기별 성능**: 간(Liver)의 경우 DSC 93.76%로 가장 높은 성능을 보였으나, 부신(Adrenal gland)이나 담낭(Gallbladder)과 같은 작은 장기들은 DSC 49%~55% 수준으로 상대적으로 낮게 측정되었다.
- **효율성**: 평균 실행 시간은 41초이며, 최대 GPU 메모리 사용량은 약 313MB 수준으로 효율적인 추론이 가능함을 입증하였다.

## 🧠 Insights & Discussion

**강점 및 기여**:
본 연구는 이미지 변환(UDA)과 자기 학습(ST)을 결합하여, 레이블이 전혀 없는 타겟 도메인에서도 높은 수준의 분할 성능을 확보할 수 있음을 보여주었다. 특히 합성 데이터의 정밀한 레이블과 실제 데이터의 도메인 특성을 동시에 활용한 미세 조정 전략이 성능 향상의 핵심 요인이었다.

**한계 및 비판적 해석**:

1. **번역 품질 의존성**: 전체 파이프라인이 초기 이미지 변환 단계의 품질에 크게 의존한다. 변환 과정에서 발생하는 아티팩트(Artifact)가 분할 모델에 전이될 위험이 있다.
2. **소형 장기 분할의 어려움**: 결과 분석에서 나타나듯, 크기가 작고 경계가 불분명한 장기들에 대해서는 여전히 낮은 정확도를 보인다. 이는 단순한 도메인 적응만으로는 해결하기 어려운 해부학적 특성의 문제일 수 있다.
3. **계산 자원**: 저자들은 실용적 적용 가능성을 언급했으나, 학습 과정에서 GAN과 3D nnU-Net을 동시에 운용해야 하므로 상당한 학습 시간(17시간)과 자원이 소모된다.

## 📌 TL;DR

DRL-STNet은 **분리 표현 학습(DRL) 기반의 이미지 변환**과 **반복적 자기 학습(ST)**을 결합하여 CT-MRI 간의 교차 모달리티 의료 영상 분할 문제를 해결한 프레임워크이다. FLARE 데이터셋에서 기존 SOTA 대비 평균 DSC를 대폭 향상시키며 그 효과를 입증하였다. 이 연구는 레이블 확보가 어려운 의료 현장에서 소스 도메인의 지식을 타겟 도메인으로 효율적으로 전이시키는 실용적인 방법론을 제시하며, 향후 확산 모델(Diffusion Models)이나 다중 모달리티 학습으로 확장될 가능성이 크다.
