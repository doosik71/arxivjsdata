# Siamese Keypoint Prediction Network for Visual Object Tracking

Qiang Li, Zekui Qin, Wenbo Zhang, and Wen Zheng (2020)

## 🧩 Problem to Solve

본 논문은 비디오 시퀀스 내에서 임의의 타겟 객체의 위치를 추정하는 Visual Object Tracking(VOT) 문제를 다룬다. 최근 Siamese 패러다임 기반의 추적기들이 높은 성능을 보이고 있으나, 기존 방식들은 크게 두 가지 문제점을 가지고 있다.

첫째, 많은 고성능 Siamese 추적기들이 복잡한 Anchor-based detection 네트워크에 크게 의존하고 있어 설계와 구현이 까다롭다. 둘째, 타겟과 유사한 외형을 가진 배경의 방해 요소(Distractors)에 취약하여 추적 성능이 급격히 저하되는 현상이 발생한다.

따라서 본 연구의 목표는 복잡한 Anchor 설계 없이도 방해 요소에 강건하며, 실시간 속도로 동작하는 Anchor-free 기반의 Siamese 추적 네트워크인 SiamKPN을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Cascade Heatmap 전략**을 통한 coarse-to-fine 예측 모델링이다. 

핵심 직관은 단계(Stage)가 진행될수록 정답 Heatmap의 분산을 점진적으로 줄여, 네트워크가 느슨한 감독(Loose supervision)에서 엄격한 감독(Strict supervision)으로 학습하게 만드는 것이다. 이를 통해 추론 과정에서 예측된 Heatmap이 단계별로 타겟에 더욱 집중되고 방해 요소에 의한 반응은 억제되어, 결과적으로 정밀한 Keypoint 예측이 가능해진다. 또한, Siamese 구조 내에서 Anchor-free 방식을 도입하여 복잡성을 줄이면서도 높은 정확도를 달성하였다.

## 📎 Related Works

논문에서는 딥러닝 기반의 추적 방법을 크게 두 가지 카테고리로 분류하여 설명한다.

1.  **Feature-Extraction Tracking**: 딥 네트워크를 특징 추출기로만 사용하고 타겟 예측은 고전적인 방식(예: SVM, Correlation Filter)에 의존하는 방식이다. ECO, UPDT 등이 이에 해당하며, 효율적인 특징 활용을 시도했지만 최근의 엔드투엔드 학습 방식에 비해 성능적 한계가 있다.
2.  **End-to-End Tracking**: 특징 추출과 타겟 예측을 하나의 통합된 네트워크로 학습하는 방식이다. MDNet, ATOM, DiMP 등이 대표적이다. 특히 ATOM과 DiMP는 IoU-Net을 통해 높은 정확도를 달성했지만, 이는 여전히 Anchor-based detection 패러다임에 속해 있어 Anchor 설정에 민감하다는 한계가 있다.

SiamKPN은 이러한 기존의 Anchor-based 방식과 달리, Heatmap 회귀(Regression)를 통해 중심점, 크기, 오프셋을 직접 예측하는 완전한 Anchor-free 방식을 채택하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
SiamKPN은 특징 학습을 위한 **Modified ResNet-50 Siamese Backbone**과 예측 모델링을 위한 **Cascade of compact KPN heads**로 구성된다.

### 1. Siamese Backbone
dense prediction 작업에 적합하도록 ResNet-50을 수정하여 사용한다.
- **Spatial Stride 감소**: $\text{conv4}_1$과 $\text{conv5}_1$ 레이어의 stride를 1로 변경하여 더 큰 해상도의 특징 맵을 유지한다.
- **Dilated Convolution**: $\text{conv4}$와 $\text{conv5}$ 블록의 dilation rate를 각각 2와 4로 설정하여 수용 영역(Receptive field)을 넓혔다.
- 최종적으로 $\text{conv3}_4, \text{conv4}_6, \text{conv5}_3$ 레이어의 출력을 추출하여 특징으로 사용하며, $1 \times 1$ convolution을 통해 채널 수를 조정한다.

### 2. Keypoint Prediction (KPN) Head
KPN 헤드는 세 개의 $3 \times 3$ convolution과 하나의 $5 \times 5$ depth-wise cross-correlation(DW-Corr)으로 구성된다. 수식으로 표현하면 다음과 같다.

$$\tilde{y}^{(s)} = \text{Corr}(\psi^{(s)}\hat{y}, \text{Conv}(\psi^{(s)}x, w^{(s)}_a))$$
$$\psi^{(s)}x = \text{Conv}(x^{(s-1)}, w^{(s)}_t)$$
$$\psi^{(s)}\tilde{y} = \text{Conv}(\tilde{y}^{(s-1)}, w^{(s)}_s)$$

여기서 $x$는 타겟 특징 맵, $\tilde{y}$는 서치 특징 맵을 의미하며, $w_t, w_s, w_a$는 각각 타겟, 서치, 내부 조정 convolution의 파라미터이다. 예측된 $\hat{y}^{(s)}$는 총 5개의 채널을 가지며 각각 다음을 예측한다.
- 중심점(Center point): 1채널
- 포인트 오프셋(Point offsets $\{\hat{o}_x, \hat{o}_y\}$): 2채널
- 타겟 크기(Target size $\{\hat{s}_h, \hat{s}_w\}$): 2채널

### 3. Cascade Heatmap Supervision
본 논문의 핵심인 Cascade 전략은 정답 Gaussian Heatmap의 분산을 단계별로 축소시키는 방식이다.

$$y^{(s)}_{ij} = \exp \left\{ -\frac{(i-i_c)^2 + (j-j_c)^2}{2(\rho^{s-1}\sigma)^2} \right\}$$

여기서 $(i_c, j_c)$는 타겟 중심점, $s$는 스테이지 번호, $\rho \in (0, 1]$는 분산 $\sigma$의 축소 강도를 조절하는 계수이다. $s$가 증가할수록 Heatmap은 중심점에 더 뾰족하게 집중되며, 이는 네트워크에 더욱 엄격한 감독 신호를 제공하여 예측을 정교하게 다듬는 역할을 한다.

### 4. 학습 목표 및 손실 함수
SiamKPN은 멀티태스크 손실 함수를 사용하여 엔드투엔드로 학습된다.
- **Keypoint Estimation Loss**: 가중치 균형을 맞춘 Focal Loss를 사용한다.
$$\mathcal{L}_{kpt} = -(1-\gamma) \sum_{i,j} \mathbb{I}[y_{ij}=1](1-\hat{y}_{ij})^\alpha \log \hat{y}_{ij} - \gamma \sum_{i,j} (1-y_{ij})^\beta (\hat{y}_{ij})^\alpha \log(1-\hat{y}_{ij})$$
- **Offsets & Size Loss**: Smooth $L_1$ loss를 사용하여 예측값과 정답 간의 차이를 최소화한다.
$$\mathcal{L}_{offs} = \sum_{ij} \ell_{\text{smooth}_1}(\hat{o}_{ij} - o_{ij}), \quad \mathcal{L}_{size} = \sum_{ij} \ell_{\text{smooth}_1}(\hat{s}_{ij} - s_{ij})$$

전체 손실 함수는 다음과 같이 정의된다.
$$\mathcal{L} = \sum_{s} \mathcal{L}^{(s)}_{kpt} + \lambda_1 \mathcal{L}^{(s)}_{offs} + \lambda_2 \mathcal{L}^{(s)}_{size}$$

### 5. 추론 및 추적 절차
온라인 업데이트 없이 오프라인 학습된 가중치만을 사용한다. 이전 프레임의 위치를 중심으로 서치 영역을 크롭하고, Cascade KPN을 통과시켜 최종 응답 맵을 얻는다. 이때 인접 프레임 간의 스케일 및 종횡비 변화에 대해 페널티 $\tau_{penalty}$를 부여하여 갑작스러운 바운딩 박스의 변화를 억제하고, 가우시안 스무딩을 통해 최종 타겟 중심점을 결정한다.

## 📊 Results

### 실험 설정
- **데이터셋**: OTB-100, VOT2018, LaSOT, GOT-10k.
- **구현**: PyTorch, GTX 1080Ti GPU 사용.
- **비교 대상**: SiamRPN++, DiMP-50, ATOM 등 최신 추적기.

### 주요 결과
1.  **OTB-100**: Success AUC 0.712, Precision 0.927을 기록하며 비교 대상 중 가장 높은 성능을 보였다. 특히 SiamRPN++ 대비 AUC에서 2.3% 향상되었다.
2.  **VOT2018**: EAO(Expected Average Overlap) 점수 0.440으로 DiMP-50과 동등한 수준의 최상위 성능을 달성했다. 주목할 점은 온라인 업데이트가 없는 모델임에도 불구하고 Robustness 점수가 다른 Siamese 추적기들보다 훨씬 높게 나타났다는 것이다.
3.  **LaSOT**: 온라인 업데이트를 수행하는 DiMP보다는 낮지만, 온라인 업데이트가 없는 모델 중에서는 가장 우수한 성능을 보였으며, ATOM보다도 높은 정밀도를 기록했다.
4.  **GOT-10k**: 온라인 업데이트가 없는 추적기 중에서는 SR 0.75 및 AO 점수에서 가장 우수한 성능을 보였다.
5.  **속도**: 3단계 구조인 SiamKPN-3s는 24 FPS로 동작하여 실시간 추적이 가능하다.

### Ablation Study
- **스테이지 수의 영향**: 스테이지가 늘어날수록(1 $\rightarrow$ 2 $\rightarrow$ 3) 성능은 향상되지만 속도는 감소한다. 하지만 3단계에서도 여전히 실시간 속도를 유지한다.
- **Variance Decay의 영향**: 분산을 고정했을 때보다 점진적으로 줄였을 때(Variance decay 적용 시) OTB-100에서 AUC와 Precision이 모두 상승하여, loose-to-strict 감독 전략의 유효성이 입증되었다.

## 🧠 Insights & Discussion

본 논문은 Anchor-free 방식의 Siamese 네트워크에 Cascade Heatmap 구조를 도입함으로써, 복잡한 Anchor 설계 없이도 매우 높은 정밀도와 강건성을 확보할 수 있음을 보여주었다. 

특히 **분산 감소 전략(Variance decay)**은 타겟과 유사한 방해 요소(Distractors)가 존재하는 환경에서 네트워크가 타겟에만 집중하도록 유도하는 효과적인 기법임이 확인되었다. 이는 단순한 모델 적층(Stacking)보다 훨씬 강력한 정제(Refinement) 효과를 제공한다.

다만, LaSOT나 GOT-10k와 같은 대규모/다양성 데이터셋에서는 DiMP와 같이 온라인 업데이트를 수행하는 모델들이 여전히 우위를 점하고 있다. 이는 처음 보는 클래스의 객체를 추적하거나 장기간 추적해야 하는 상황에서는 온라인 적응(Online adaptation)이 필수적임을 시사한다.

## 📌 TL;DR

SiamKPN은 Anchor-free 기반의 Siamese 추적기로, 단계별로 정답 Heatmap의 분산을 줄여가는 **Cascade Heatmap 전략**을 통해 타겟 정밀도를 높이고 방해 요소에 대한 강건성을 확보했다. 온라인 업데이트 없이도 최신 Siamese 추적기들을 압도하는 성능을 보이며 실시간(24 FPS)으로 동작한다. 이 연구는 Anchor-free 방식이 Siamese 추적에서도 충분히 경쟁력이 있음을 입증했으며, 향후 Anchor-free 기반의 실시간 고성능 추적기 연구에 중요한 이정표가 될 것으로 보인다.