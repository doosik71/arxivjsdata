# NudgeSeg: Zero-Shot Object Segmentation by Repeated Physical Interaction

Chahat Deep Singh, Nitin J. Sanket, Chethan M. Parameshwara, Cornelia Fermüller, Yiannis Aloimonos (2021)

## 🧩 Problem to Solve

본 논문은 정해진 클래스나 훈련 데이터에 의존하지 않고, 처음 보는 물체(zero-shot objects)들이 뒤섞여 있는 Cluttered scene에서 개별 물체를 분할(Segmentation)하는 문제를 해결하고자 한다.

기존의 딥러닝 기반 객체 분할 방식은 훈련 단계에서 사용된 클래스와 객체의 수에 따라 성능이 결정되므로, 학습 데이터에 포함되지 않은 새로운 객체에 대해서는 일반화 성능이 떨어진다는 한계가 있다. 또한, 단순한 이미지 프레임 기반의 분할은 인식 및 패턴 매칭 큐(cue)에 의존하기 때문에 복잡한 환경에서 한계가 명확하다. 본 연구의 목표는 로봇의 '능동적(active)' 성질과 환경과의 '상호작용(interaction)' 능력을 활용하여, 추가적인 기하학적 제약 조건을 생성함으로써 zero-shot 샘플에 대한 분할 성능을 높이는 프레임워크인 NudgeSeg를 제안하는 것이다.

## ✨ Key Contributions

NudgeSeg의 핵심 아이디어는 단색 단안 카메라(monochrome monocular camera)만을 사용하여 물체를 반복적으로 '밀어내는(nudging)' 행위를 통해 모션 큐(motion cues)를 얻고, 이를 통해 분할 마스크를 정교화하는 것이다.

주요 기여 사항은 다음과 같다:

1. **능동적-상호작용적 넛징 프레임워크(Active-Interactive Nudging Framework)**: 단색 단안 카메라만을 이용하여 zero-shot 객체를 분할하는 NudgeSeg 프레임워크를 제안하였다.
2. **광학 흐름의 불확실성 활용**: Optical flow의 불확실성을 개념화하여 물체들이 쌓여 있는 더미(object clutter pile)의 위치를 찾아내는 방법을 제시하였다.
3. **로봇 플랫폼 독립성 증명**: 쿼드로터(quadrotor)와 로봇 팔(UR-10) 등 서로 다른 구조의 로봇을 사용하여 실험함으로써, 제안하는 프레임워크가 로봇의 하드웨어 구조에 구애받지 않고 작동함을 입증하였다.

## 📎 Related Works

논문에서는 관련 연구를 세 가지 분야로 나누어 설명한다:

1. **인스턴스 및 시맨틱 분할(Instance and Semantic Segmentation)**: Fully Convolutional Neural Networks (FCN), Mask R-CNN, PointRend 등이 언급된다. 이러한 방식들은 높은 정확도를 보이지만, 학습되지 않은 새로운 클래스의 객체를 분할하는 zero-shot 상황에서는 성능이 급격히 저하된다.
2. **모션 분할(Motion Segmentation)**: 3D 모션이나 Optical flow, 이벤트 카메라(event camera)를 이용한 분할 방식들이 연구되었다. 특히 0-MMS와 같은 방식은 이벤트 카메라를 통해 독립적으로 움직이는 객체를 분할하지만, 단일한 큰 움직임에 의존한다는 차이가 있다.
3. **능동적 및 상호작용적 접근법(Active and Interactive Approaches)**: 로봇이 정보를 더 많이 수집하기 위해 탐색적으로 움직이거나(Active vision), 환경과 상호작용하여 인지 문제를 단순화하는 접근법이다. 기존 연구들은 주로 Depth 센서나 스테레오 카메라를 사용하였으나, NudgeSeg는 단색 단안 카메라만을 사용한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

NudgeSeg는 크게 **능동적 인지(Active perception)** 단계와 **상호작용적 인지(Interactive perception)** 단계로 구성된다.

### 1. 능동적 인지 (Active Perception)

객체 분할의 전제 조건은 물체 더미의 위치를 찾는 것이다. Depth 센서가 없으므로, 로봇의 카메라를 움직여 Depth와 상관관계가 있는 함수를 얻는다.

- **불확실성 추정**: 두 프레임 사이의 Optical flow 추정 문제에서 발생하는 불확실성 $\rho$(Heteroscedastic Aleatoric uncertainty)를 계산한다. 이 $\rho$ 값은 전경(foreground)과 배경(background)의 경계와 상관관계가 있다.
- **첫 번째 넛징 지점($N_1$) 결정**: $\rho$ 값에 대해 형태학적 연산(morphological operations)을 수행하여 블롭(blob)들을 찾고, 그 중 평균 불확실성이 가장 높은 블롭 $O^\kappa$를 선택한다. 이후, 해당 블롭의 볼록 껍질(convex hull) $C(O^\kappa)$에서 중심점 $(O^\kappa)$과 가장 먼 지점을 첫 번째 넛징 지점으로 설정한다.
$$N_1 = \text{argmax}_x \|C(O^\kappa)_x - (O^\kappa)\|^2$$

### 2. 상호작용적 인지 (Interactive Perception)

물체를 실제로 밀어내어 발생하는 움직임을 통해 강체(rigid) 부분을 클러스터링한다.

- **클러스터링 (DBSCAN)**: Optical flow를 이용하여 장면을 분할한다. DBSCAN 알고리즘을 사용하며, 입력 데이터는 $[x, M, A]^T$ (이미지 좌표, flow 크기 $M$, flow 각도 $A$)의 4차원 데이터이다. 다음 조건들을 만족할 때 같은 클러스터로 묶는다:
  - 거리 조건: $\|X-Y\|^2 < \tau_d$
  - 크기 조건: $\|M_X - M_Y\|^2 < \tau_M$
  - 각도 조건: $\min(|A_X - A_Y|, 2\pi - |A_X - A_Y|) \le \tau_A$
- **넛징 정책 (Nudging Policy)**: 각 클러스터 $H_i^k$의 Optical flow 공분산 행렬 $\Sigma_i^k$를 계산한다.
$$\Sigma_i^k = E((\dot{p}_i^k - E(\dot{p}_i^k))(\dot{p}_i^k - E(\dot{p}_i^k))^T)$$
여기서 고윳값(eigenvalues) $\lambda_{max}, \lambda_{min}$을 통해 조건수(condition number) $\kappa = |\lambda_{max}|/|\lambda_{min}|$를 구한다. 가장 정보가 많은(이미 잘 분할된) 객체를 제외하고, 두 번째로 $\kappa$가 큰 클러스터를 선택하여 그 고유벡터 $v_{min}$ 방향으로 물체를 민다.
- **마스크 정교화 (Mask Refinement)**: 넛징 후 Optical flow를 이용해 마스크를 새로운 프레임으로 워핑(warping)하여 전파하고, 기존 마스크와 합집합($\cup$) 연산을 통해 업데이트한다.

### 3. 검증 및 종료 (Verification and Termination)

연속된 반복에서 평균 IoU 변화가 임계값보다 작으면 검증 단계를 거친다. 각 세그먼트를 기하학적 중심 방향으로 밀어내어 더 이상 쪼개지지 않는지 확인한 후 프로세스를 종료한다.

### 4. 네트워크 세부 사항

Optical flow $\dot{p}$는 PWC-Net 기반의 CNN을 사용한다. 특히, 불확실성 $\rho$를 함께 추정하기 위해 출력 채널을 2개에서 4개로 확장하였으며, 다음과 같은 손실 함수 $L$을 사용하여 자기지도 학습(self-supervised) 방식으로 훈련하였다.
$$L = \sum_{\forall l} \alpha_l E_x \left( \frac{\|\hat{\dot{p}}_l(x) - \tilde{\dot{p}}_l(x)\|_1}{\log_e(1 + e^{(\rho_l(x) + \epsilon)})} \right) + \sum_{\forall l} \alpha_l E_x (\log_e(1 + e^{\rho_l(x)}))$$
여기서 $l$은 피라미드 레벨, $\epsilon$은 수치적 안정성을 위한 정규화 값($10^{-3}$)이다.

## 📊 Results

### 실험 설정

- **하드웨어**: UR10 로봇 팔, PRGLabrador 500$\alpha$ 쿼드로터. 단색 단안 카메라(800x600px, 30fps) 사용.
- **데이터셋**: GrassMoss, Rocks, YCB, YCB-attached(물체들이 붙어 있는 adversarial sample)의 4가지 시퀀스.
- **비교 대상**: 0-MMS (이벤트 카메라 기반), Mask-RCNN, PointRend (패시브 분할).
- **지표**: Intersection over Union (IoU), Detection Rate ($DR_{50}, DR_{75}$). $DR$은 $IoU \ge \tau$일 때 성공으로 간주한다.

### 주요 결과

- **Zero-shot 성능**: NudgeSeg는 zero-shot 객체들에 대해 평균 86% 이상의 높은 검출률(DR)을 보였다.
- **능동적 vs 패시브**: GrassMoss와 Rocks 같은 zero-shot 샘플에서 패시브 방식(Mask-RCNN, PointRend)은 매우 낮은 성능을 보인 반면, NudgeSeg는 압도적인 성능 차이를 보였다. 이는 능동적 상호작용이 인식되지 않은 객체를 분할하는 데 필수적임을 시사한다.
- **모션 큐의 영향**: 단일 넛징에 의존하는 0-MMS보다 반복적 넛징을 수행하는 NudgeSeg의 성능이 더 우수했다.
- **강건성 분석**: Optical flow의 크기($M$)보다 각도($A$)에 노이즈를 주었을 때 성능이 훨씬 더 급격하게 저하됨을 확인하였다. 이는 능동적 분할에서 flow의 방향 정보가 매우 중요함을 의미한다.

## 🧠 Insights & Discussion

본 논문은 인지(Perception)와 상호작용(Interaction)의 시너지를 통해 복잡한 인지 문제를 단순한 제어 문제로 치환할 수 있음을 보여준다. 특히 Depth 센서 없이 단색 카메라만으로 zero-shot 분할을 구현했다는 점이 고무적이다.

**강점 및 한계**:

- **강점**: 로봇의 구조에 상관없이 적용 가능하며, 학습 데이터에 없는 객체에 대해서도 실용적인 분할 성능을 제공한다.
- **한계**: Optical flow의 정확도에 매우 의존적이다. Flow 계산에 오차가 발생하면 분할 성능이 급격히 떨어진다. 또한, 현재는 2D 평면 위에서의 상호작용만을 가정하고 있다.

**비판적 해석**:
논문은 Optical flow의 각도 오차에 민감하다는 점을 들어 향후 네트워크 경량화 및 최적화가 필요하다고 주장한다. 하지만 실제 환경에서는 단순히 네트워크 속도를 높이는 것보다, 조명 변화나 가려짐(occlusion)과 같은 환경적 노이즈에 강건한 Optical flow 추정 기법을 결합하는 것이 더 시급한 과제일 것으로 판단된다.

## 📌 TL;DR

NudgeSeg는 단색 단안 카메라를 장착한 로봇이 물체를 반복적으로 밀어내어 발생하는 **Optical flow(모션 큐)를 통해 학습되지 않은 객체(zero-shot)를 분할**하는 프레임워크이다. 능동적 인지로 물체 더미를 찾고, 상호작용적 인지로 개별 객체를 분리하며 마스크를 정교화한다. 실험 결과 zero-shot 객체에 대해 86% 이상의 높은 검출률을 보였으며, 이는 기존의 패시브한 딥러닝 방식보다 월등히 우수하다. 이 연구는 향후 로봇이 새로운 환경에서 스스로 물체를 배우는 **평생 학습(lifelong learning)** 시스템의 기초 단계로 활용될 가능성이 높다.
