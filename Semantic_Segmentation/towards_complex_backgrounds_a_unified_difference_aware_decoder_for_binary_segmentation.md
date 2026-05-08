# Towards Complex Backgrounds: A Unified Difference-Aware Decoder for Binary Segmentation

Jiepan Li, Wei He, and Hongyan Zhang (2022)

## 🧩 Problem to Solve

본 논문은 이미지 내에서 관심 대상(foreground)을 배경(background)으로부터 분리하는 이진 분할(Binary Segmentation) 문제, 특히 배경이 복잡한 상황에서의 분할 성능 향상을 목표로 한다. 이진 분할은 돌출 객체 검출(Salient Object Detection, SOD), 위장 객체 검출(Camouflaged Object Detection, COD), 폴립 분할(Polyp Segmentation), 거울 검출(Mirror Detection) 등 다양한 작업에 적용된다.

기존의 디코더 설계 방식은 특정 객체의 특성에 맞추어 설계되었거나, 인코더(backbone)가 추출한 다단계 특징을 단순히 결합하는 방식에 치중해 왔다. 하지만 전경과 배경의 시각적 차이가 매우 적은 위장 객체나, 배경에 노이즈가 많은 복잡한 환경에서는 기존 방식들이 객체의 경계를 정확히 구분하지 못하고 흐릿한 결과를 내놓는 한계가 있다. 따라서 본 연구의 목표는 다양한 배경 환경에서도 범용적으로 적용 가능하며, 전경과 배경의 차이를 명확하게 극대화할 수 있는 통합 디코더 구조를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간의 시각 시스템이 객체를 인식하는 과정을 모방한 **Difference-Aware Decoder (DAD)** 구조를 제안하는 것이다. 인간의 눈은 먼저 관심 대상에 대해 대략적인 위치를 파악하고, 이후 객체 주변의 배경 영역에 주의를 기울이며, 최종적으로 뇌에서 전경과 배경의 차이를 확대하여 정밀한 외곽선을 인식한다.

이를 위해 DAD는 다음의 세 단계 과정을 거친다.

1. **Stage A**: 전경의 대략적인 위치 정보를 담은 가이드 맵(Guide Map)을 생성한다.
2. **Stage B**: 중위 레벨 특징들을 융합하여 배경 인식 특징(Background-aware features)을 추출한다.
3. **Stage C**: 가이드 맵과 배경 인식 특징을 융합하여 전경과 배경의 차이를 명시적으로 확대함으로써 최종 분할 결과를 도출한다.

## 📎 Related Works

논문에서는 주로 SOD와 COD 두 가지 작업의 발전 과정을 다룬다.

- **Salient Object Detection (SOD)**: 초기에는 CNN을 이용해 전역 및 지역 문맥을 모델링하여 돌출 영역을 찾았으며, 이후 Encoder-Decoder 구조가 주류가 되었다. 다양한 디코더가 제안되었으나, 대부분 단순한 배경에서는 잘 작동하지만 복잡한 배경에서는 성능이 저하되는 문제가 있었다.
- **Camouflaged Object Detection (COD)**: SOD보다 훨씬 어려운 작업으로, 자연계의 포식자가 먹이를 찾는 과정을 모방하여 검색 및 식별 과정을 설계한 모델(예: SINet)들이 제안되었다. 그러나 이러한 방식들은 주로 전경 객체 자체에 집중하며 배경이 주는 정보의 기여도를 간과하는 경향이 있다.
- **Dual-branch Decoder**: 일부 연구에서 두 개의 브랜치를 사용하여 특징을 정교화하거나 상호 보완적인 작업을 수행하는 구조를 제안했다. 본 논문은 이를 확장하여 한 브랜치는 전경(coarse foreground)을, 다른 브랜치는 배경(background-aware)을 탐색하게 하여 두 정보의 차이를 이용하는 전략을 취한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

DAD는 입력 이미지에서 Backbone(VGG, ResNet, Res2Net, PVT 등)을 통해 다단계 특징을 추출하고, 이를 세 가지 스테이지(Stage A, B, C)를 통해 처리하여 최종 마스크를 생성한다.

### 2. Stage A: Guide Map Generator (GMG)

전경의 대략적인 위치를 파악하는 단계이다.

- **Field Expansion Module (FEM)**: 수용 영역(Receptive Field)을 효율적으로 확장하기 위해 4개의 병렬 경로를 사용한다. 특히 $3 \times 3$ 합성곱 층의 dilation rate를 $\{4, 8, 16, 32\}$ 및 $\{2, 4, 8, 16\}$으로 순차적으로 증가시켜 매우 넓은 영역의 문맥을 포착한다.
- **Dual Residual Attention (DRA)**: FEM의 출력에 DRA 모듈을 적용하여 전역적인 의존성을 캡처하고 배경의 간섭을 줄인다.
- 이후, 이 고수준 특징을 저수준(lowest-level) 특징과 결합하여 세부 정보를 보완한 뒤 가이드 맵 $C_0$를 생성한다.

### 3. Stage B: Middle Feature Fusion (MFF)

배경 정보를 효과적으로 추출하기 위해 인코더의 중위 레벨 특징 3개를 융합한다.

- **융합 전략**: 특징 맵들을 무조건 가장 크거나 작은 사이즈로 맞추는 대신, 중간 사이즈를 기준으로 상향/하향 샘플링하여 크기를 맞춘다.
- 이 과정에서 FEM을 적용하여 배경에 대한 풍부한 문맥 정보를 확보하고, 최종적으로 배경 인식 특징을 생성한다.

### 4. Stage C: Difference-Aware Extractor (DAE)

가이드 맵과 배경 인식 특징을 융합하여 전경-배경 차이를 극대화하는 단계이다.

**A. Difference Guidance Module (DGM)**
가이드 맵 $M$과 배경 인식 특징 $F$를 입력으로 받는다.

- 먼저 두 특징의 크기를 맞춘 뒤, Cross-Attention 메커니즘을 통해 전경과 배경의 관계를 확률 값으로 계산한다.
- $$R_{ij} = \frac{\exp(Q_i \times G_j)}{\sum_{j=1}^{C} \exp(Q_i \times G_j)}$$
- 이를 통해 강화된 특징 $E = \beta \times A + F_0$ (여기서 $\beta$는 학습 가능 파라미터)를 얻는다.

**B. Difference Enhancement Module (DEM)**
가이드 맵을 통해 전경 확률 $P$와 배경 확률 $1-P$를 구분하여 특징을 분리한다.

- 전경 특징: $D_f = P \times E$
- 배경 특징: $D_b = (1-P) \times E$
- 최종적으로 두 특징의 차이를 학습 가능한 파라미터 $\theta, \epsilon$을 이용해 융합한다.
- $$D = \theta \times D_f - \epsilon \times D_b$$
이 연산을 통해 전경과 배경의 차이가 명시적으로 확대된 특징 $D$가 생성된다.

### 5. 학습 절차 및 손실 함수

DAE 모듈은 두 번 반복 적용되어 $C_1$과 $C_2$를 생성한다. 최종적으로 $C_0, C_1, C_2$ 세 개의 맵을 생성하며, 가중치 적용 이진 교차 엔트로피(Weighted BCE) 손실과 가중치 적용 IoU 손실을 합산하여 사용한다.
$$\text{loss} = \sum_{i=0}^{2} (\text{loss}_{wbce}(C_i, GT) + \text{loss}_{wiou}(C_i, GT))$$

## 📊 Results

### 1. 실험 설정

- **대상 작업**: SOD, COD, 폴립 분할, 거울 검출 4가지 작업.
- **Backbones**: VGG-16, ResNet-50, Res2Net-50, PVT-v2-b2.
- **평가 지표**: $S_\alpha$ (Structure-measure), $E_\phi$ (E-measure), $F^w_\beta$ (Weighted F-measure), $M$ (MAE), 그리고 폴립 분할의 경우 Dice와 IoU를 추가 사용한다.

### 2. 주요 결과

- **SOD**: 6개 데이터셋에서 평가한 결과, 모든 Backbone에서 DAD가 기존 SOTA 방법들보다 높은 성능을 보였다. 특히 PVT-v2-b2와 결합했을 때 최적의 성능을 달성했다.
- **COD**: 위장 객체 검출에서도 ResNet, Res2Net, PVT 등 모든 백본에 대해 SINet V2 및 DTIT보다 우수한 성능을 기록했다.
- **폴립 분할 및 거울 검출**: ClinicDB, ColonDB, ETIS 및 MSD 데이터셋에서 각각 기존 방법들(PraNet, Polyp-PVT, MirrorNet 등)을 상회하는 정량적 수치를 보였다.

### 3. 분석 및 어블레이션 연구

- **FEM의 효용성**: ASPP, RFB, DenseASPP보다 FEM이 수용 영역 확장 및 성능 향상에 더 효과적임을 확인했다.
- **DGM 및 DEM의 영향**: DGM을 제거하거나 DEM에서 전경-배경 차이($F-B$) 대신 단일 특징($F$ 또는 $B$)만 사용했을 때 성능이 크게 하락했다. 이는 전경과 배경의 차이를 명시적으로 학습하는 구조가 핵심임을 시사한다.
- **DAE 반복 횟수**: DAE를 2회 반복 적용했을 때 가장 좋은 성능을 보였으며, 3회 이상 반복 시 오히려 성능이 저하되었다.

## 🧠 Insights & Discussion

본 논문의 강점은 특정 작업에 국한되지 않고 이진 분할이라는 일반적인 문제에 접근하여, 인간의 시각적 인지 과정을 공학적으로 잘 구현했다는 점이다. 특히 전경 가이드 맵과 배경 인식 특징을 분리하여 추출한 뒤, 이를 '차이(Difference)' 관점에서 융합하는 방식은 배경이 복잡한 데이터셋에서 매우 강력한 성능을 발휘한다.

또한, 특정 백본에 종속되지 않고 VGG부터 Transformer 기반의 PVT까지 다양한 인코더에 적용 가능함을 입증함으로써 제안한 디코더 구조의 범용성과 전이 가능성(Transferability)을 증명하였다.

다만, 논문에서 명시적으로 언급되지는 않았으나, 듀얼 브랜치 구조와 반복적인 DAE 적용으로 인해 단일 브랜치 디코더 대비 연산량과 메모리 사용량이 증가했을 가능성이 있다. 또한, $C_0, C_1, C_2$ 세 단계의 맵을 모두 지도 학습하는 방식이 학습 시간을 증가시켰을 것으로 추측된다.

## 📌 TL;DR

본 연구는 인간의 눈이 객체를 인식하는 방식을 모방하여, 전경 가이드 맵 생성과 배경 인식 특징 추출을 독립적으로 수행하고 그 차이를 극대화하는 **Difference-Aware Decoder (DAD)**를 제안하였다. 이 구조는 SOD, COD, 의료 영상(폴립), 거울 검출 등 배경이 복잡한 다양한 이진 분할 작업에서 기존 SOTA 모델들을 능가하는 성능을 보였으며, 다양한 백본 네트워크에 범용적으로 적용 가능하다는 것을 입증하였다. 향후 도로 추출과 같은 다른 이진 분할 작업으로의 확장이 기대되는 연구이다.
