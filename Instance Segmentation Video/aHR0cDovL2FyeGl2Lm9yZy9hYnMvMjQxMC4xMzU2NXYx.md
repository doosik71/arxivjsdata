# SDI-Paste: Synthetic Dynamic Instance Copy-Paste for Video Instance Segmentation

Sahir Shrestha, Weihao Li, Gao Zhu, Nick Barnes (2024)

## 🧩 Problem to Solve

본 논문은 비디오 인스턴스 분할(Video Instance Segmentation, VIS) 모델 학습을 위한 고품질 데이터 확보의 어려움을 해결하고자 한다. VIS는 비디오 시퀀스 전체에서 객체의 인식(Recognition), 분할(Segmentation), 그리고 추적(Tracking)을 동시에 수행해야 하는 복잡한 작업이다.

이 작업을 위한 학습 데이터셋을 구축하기 위해서는 모든 프레임에 대해 객체의 정교한 분할 마스크(Segmentation Mask)를 생성해야 하는데, 이는 단순한 분류나 검출 작업에 비해 인간 작업자의 비용과 시간이 훨씬 더 많이 소요되는 심각한 병목 현상을 초래한다. 또한, 희귀 객체 카테고리의 경우 실제 영상 데이터를 수집하는 것 자체가 매우 어렵다는 문제가 있다. 기존의 Copy-Paste 방식은 기존 데이터셋 내의 객체를 재사용하거나 3D 모델을 활용했으나, 이는 확장성(Scalability)이 떨어지거나 자원 소모가 크다는 한계가 있었다. 따라서 본 연구의 목표는 생성형 모델을 활용하여 수동 레이블링 없이도 무한히 확장 가능한 합성 동적 인스턴스 생성 및 증강 파이프라인을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최신 Text-to-Video(T2V) 생성 모델의 발전된 화질을 이용하여, 실제 세계의 객체가 가진 **시간적 역동성(Temporal Dynamism)**을 반영한 합성 객체를 생성하고 이를 기존 비디오 배경에 삽입하는 것이다.

단순히 정지된 이미지의 객체를 반복적으로 붙여넣는 것이 아니라, 시간에 따라 형태가 변하고 움직이는 '동적 인스턴스(Dynamic Instance)'를 생성하여 VIS 모델이 객체의 외형 변화와 움직임을 더 잘 학습하도록 유도한다. 이를 위해 T2V 생성 $\rightarrow$ 자동 분할 $\rightarrow$ 필터링 $\rightarrow$ 동적 경로 기반 합성으로 이어지는 **SDI-Paste** 파이프라인을 제안한다.

## 📎 Related Works

**1. Video Instance Segmentation (VIS):**
VIS 방법론은 크게 Offline과 Online 방식으로 나뉜다. Offline 방식은 비디오 전체 프레임을 동시에 처리하여 문맥 이해도가 높지만 메모리 소모가 크다. 반면, Online 방식은 국소 범위의 프레임만 사용하여 실시간 처리가 가능하지만 성능이 상대적으로 낮다. 본 논문은 실용적인 학습/추론 효율성을 위해 CTVIS와 IDOL 같은 Online VIS 네트워크를 베이스라인으로 사용한다.

**2. Video Generation:**
최근 Diffusion 모델 기반의 T2I(Text-to-Image) 및 T2V(Text-to-Video) 모델이 급격히 발전했다. 특히 AnimateDiff는 Stable Diffusion 백본에 모션 모듈을 추가하여 텍스트 프롬프트 기반으로 자연스러운 움직임을 가진 비디오 클립을 생성할 수 있다.

**3. Data Augmentation:**
이미지 영역에서는 Copy-Paste 방식이 효과적임이 입증되었으며, 최근 X-Paste는 Stable Diffusion으로 생성한 합성 이미지를 사용하여 확장성을 높였다. 비디오 영역에서도 다양한 증강 기법이 연구되었으나, 대부분 기존 레이블링된 영상을 변형하는 방식이었으며 VIS 작업을 위해 합성 동적 인스턴스를 활용한 연구는 본 논문이 처음이다.

## 🛠️ Methodology

SDI-Paste 파이프라인은 크게 세 단계로 구성된다.

### 1. Synthetic Video Generation (합성 비디오 생성)

AnimateDiff 모델을 사용하여 특정 객체 클래스가 포함된 짧은 비디오 클립(16프레임)을 생성한다. 이때 다양성을 확보하기 위해 "moving", "dynamic", "changing background"와 같은 일반적인 형용사를 포함한 프롬프트를 사용한다.

- **프롬프트 예시:** *"A close up video of one moving dynamic [object] in changing background, moving camera, centred."*
- 이 과정에서 생성된 객체에 일부 형태적 오류(예: 토끼의 귀가 3개인 경우)가 발생할 수 있는데, 저자들은 이러한 변형이 오히려 네트워크가 객체의 형태 변화를 추적하는 능력을 향상시키는 챌린지로 작용하여 성능 향상에 기여한다고 분석한다.

### 2. Video Instance Segmentation (비디오 인스턴스 분할)

생성된 비디오에서 배경을 제거하고 객체만 추출하는 과정이다.

- **분할:** self-supervised transformer 기반의 TokenCut 알고리즘을 사용하여 각 프레임의 돌출 객체(Salient Object) 마스크를 추출한다.
- **필터링:** 생성된 객체가 프롬프트와 일치하는지 확인하기 위해 CLIP 모델을 사용하여 관련성 점수를 측정하고, 임계값(0.21) 미만인 경우 제거한다. 또한, 이미지 면적의 5% 미만 또는 95% 이상을 차지하는 마스크는 오분할로 간주하여 삭제한다.

### 3. Dynamic Instance Composition (동적 인스턴스 합성)

추출된 동적 인스턴스를 실제 비디오 배경($F_1, F_2, ..., F_{N_f}$)에 무작위로 배치한다.

- **객체 수 및 시작점:** 도입할 객체 수 $N_i$를 $[1, 20]$ 사이에서 무작위로 정하고, 첫 프레임의 시작 좌표 $(x_0, y_0)$를 배경 크기 $W, H$ 내에서 균등 분포로 샘플링한다.
  $$x_0 \sim U[0, W], y_0 \sim U[0, H]$$

- **선형-무작위 궤적(Linear-Random Trajectory):** 객체의 이동 경로를 설정한다. 모든 프레임에서 이동 방향(각도 $\theta$)은 일정하게 유지하지만, 프레임 간 이동 거리(변위 $\Delta$)는 무작위로 변경한다.
  $$\theta_{ij} \sim [0, 360]^\circ, \quad \Delta_{ij} \sim U[0, \Delta_{max}]$$
  이를 통해 픽셀 단위 변위 $(\delta x_{ij}, \delta y_{ij})$를 계산한다.
  $$(\delta x_{ij}, \delta y_{ij}) = (\Delta_{ij} * \cos \theta_{ij}, \Delta_{ij} * \sin \theta_{ij})$$
  최종 좌표는 이전 프레임의 좌표에 변위를 더해 결정한다.
  $$(x_{ij}, y_{ij}) = (x_{i, j-1} + \delta x_{ij}, y_{i, j-1} + \delta y_{ij})$$

- **크기 조정:** 각 카테고리별 객체 크기의 평균 $\mu_C$와 분산 $\sigma_C^2$를 기반으로 가우시안 분포에서 스케일 $S_i$를 샘플링하여 적용한다.

## 📊 Results

### 실험 설정

- **데이터셋:** Youtube-VIS 2021 (YTVIS21)
- **베이스라인 네트워크:** CTVIS, IDOL (Online VIS 모델)
- **평가 지표:** $AP, AP_{50}, AP_{75}, AR_1, AR_{10}$

### 주요 결과

- **정량적 성능 향상:** SDI-Paste를 적용했을 때, CTVIS는 $+2.9 \text{ AP} (6.5\%)$, IDOL은 $+2.1 \text{ AP} (4.9\%)$의 성능 향상을 보였다.
- **기존 증강 기법과의 비교:** 일반적인 Copy-Paste(실제 데이터 사용)나 X-Paste(합성 정지 이미지 사용)보다 SDI-Paste가 더 높은 성능을 기록했다. 특히 X-Paste 대비 $+0.4 \text{ AP}$의 추가 향상을 보였는데, 이는 동적 인스턴스가 정적 인스턴스보다 다양한 시점과 형태 변형 정보를 제공하기 때문으로 분석된다.

### 절제 연구 (Ablation Study)

- **궤적 시스템:** Bezier 곡선보다 선형(Linear) 및 선형-무작위(Linear-random) 방식이 더 효과적이었으며, 그중 선형-무작위 방식이 가장 좋은 성능을 보였다.
- **분할 방법:** X-Paste의 CLIP 가이드 방식보다 TokenCut 단독 사용 시 성능이 더 높았다.
- **인스턴스 수:** 생성된 동적 인스턴스 시퀀스의 수를 150개에서 470개로 늘렸을 때 성능이 $3.5\%$ 향상되어, 데이터의 양적 확장이 성능에 긍정적인 영향을 줌을 확인했다.

## 🧠 Insights & Discussion

본 연구는 생성형 모델을 통해 얻은 합성 데이터가 실제 VIS 모델의 성능을 유의미하게 끌어올릴 수 있음을 입증하였다. 특히 주목할 점은 생성 모델의 불완전함에서 오는 **'형태적 이상 현상(Aberrations)'**이 오히려 모델의 강건성(Robustness)을 높이는 정규화 효과를 낸다는 가설이다. 이는 실제 환경에서 객체가 일부 가려지거나 형태가 왜곡되는 상황을 시뮬레이션하는 효과를 주기 때문으로 해석된다.

다만, 본 논문은 Online VIS 모델만을 대상으로 테스트하였으며, T2V 모델의 생성 속도와 계산 비용으로 인해 더 대규모의 데이터셋을 구축하여 실험하지 못한 한계가 있다. 또한, 제안한 선형-무작위 궤적이 실제 비디오의 모든 움직임을 대변하기에는 단순하다는 점이 지적될 수 있다.

## 📌 TL;DR

SDI-Paste는 T2V 생성 모델(AnimateDiff)과 자동 분할 도구(TokenCut)를 결합하여, 수동 레이블링 없이 **시간적으로 역동적인 합성 객체**를 생성하고 이를 비디오 데이터셋에 삽입하는 증강 파이프라인이다. 실험 결과, 단순한 정지 이미지 합성보다 동적 인스턴스를 활용하는 것이 VIS 성능 향상에 훨씬 효과적임을 확인하였다. 이 방법론은 데이터 부족 문제를 해결하고 희귀 객체에 대한 학습 능력을 높일 수 있어, 향후 다양한 비디오 분석 작업으로 확장될 가능성이 높다.
