# Multiple Object Tracking with Kernelized Correlation Filters in Urban Mixed Traffic

Yuebin Yang, Guillaume-Alexandre Bilodeau (2017)

## 🧩 Problem to Solve

본 논문은 도시의 혼잡한 교통 환경(Urban Mixed Traffic)에서 발생하는 Multiple Object Tracking(MOT) 문제를 해결하고자 한다. 교통 감시 시스템에서 객체 추적은 정체 해소 및 도로 안전 평가를 위해 매우 중요하지만, 기존의 지능형 교통 시스템(ITS)은 다음과 같은 환경적 변수에 취약한 한계를 보인다.

- **폐색(Occlusion):** 차량이나 보행자가 서로 겹쳐 보일 때 객체를 구분하지 못하는 문제.
- **환경 변화:** 조명 변화, 모션 블러(Motion Blur) 등으로 인한 검출 성능 저하.
- **객체 다양성:** 도시 도로에는 자동차, 자전거, 보행자, 트럭 등 형태와 크기가 다양한 객체들이 혼재되어 있어, 특정 클래스에 최적화된 사전 학습된 검출기(Pre-trained Detector)만으로는 모든 객체를 누락 없이 검출하기 어렵다.

따라서 본 연구의 목표는 모델 프리(Model-free) 방식의 배경 제거(Background Subtraction)와 강력한 시각적 추적기인 KCF(Kernelized Correlation Filters)를 결합하여, 복잡한 데이터 연관(Data Association) 과정 없이도 폐색 및 분절(Fragmentation) 문제에 강건한 MOT 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **배경 제거(Background Subtraction)의 '검출 능력'과 KCF 추적기의 '추적 유지 능력'을 상호 보완적으로 결합**하는 것이다.

- **상호 보완적 의사결정:** 배경 제거는 새로운 객체 탐지와 스케일(Scale) 정보를 제공하며, KCF는 객체가 폐색되어 배경 제거 결과가 하나로 뭉쳐질 때 개별 객체의 위치를 추적하는 역할을 수행한다.
- **단순화된 데이터 연관:** 복잡한 merge-split 알고리즘이나 정교한 데이터 연관 기법 대신, KCF의 내부 모델과 배경 제거로 생성된 Blob 간의 중첩도(Overlap)를 이용해 상태를 결정함으로써 연산 효율성을 높였다.
- **폐색 처리 방식의 개선:** 폐색 발생 시 객체들을 그룹으로 묶어 관리하고, KCF를 통해 개별적으로 추적함으로써 폐색 해제 후의 ID 유지 능력을 향상시켰다.

## 📎 Related Works

교통 환경의 객체 추적 연구는 크게 두 가지 접근 방식으로 나뉜다.

1. **Optical Flow 기반 방식:** KLT 추적기 등을 사용하여 특징점의 움직임을 추적한다. 하지만 인접한 객체가 유사한 속도로 이동할 경우 하나로 병합되거나, 객체가 정지했을 때 특징점 흐름이 끊겨 궤적이 분절되는 한계가 있다.
2. **Background Subtraction 기반 방식:** 배경과 전경을 분리하여 Blob을 추출하는 방식이다. 모든 객체를 검출할 수 있다는 장점이 있지만, 객체 간 폐색이 발생하면 여러 객체가 하나의 Blob으로 병합되는 문제가 발생한다. 기존 연구들은 이를 해결하기 위해 복잡한 merge-split 패러다임이나 유한 상태 머신(FSM)을 도입하여 사후적으로 분리하는 방식을 사용했다.

본 논문은 이러한 기존 방식들과 달리, **KCF라는 강력한 시각적 추적기를 개별 객체에 할당**함으로써 폐색 상황에서도 객체를 개별적으로 추적하며, 사후적인 백트래킹(Backtracking) 없이 실시간으로 문제를 해결한다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인

시스템은 크게 **전경 검출(Foreground Detection) $\rightarrow$ Blob 분석(Blob Analysis) $\rightarrow$ 객체 추적(Object Tracking)**의 순서로 구성된다.

### 1. 전경 검출 및 Blob 분석

- **전경 검출:** LOBSTER 알고리즘 등을 사용하여 전경 Blob($B_i$)을 추출한다.
- **Blob 분석:** 추출된 $B_i$에 대해 미디언 필터링, Closing, Hole filling 등 형태학적 연산을 적용한다. 이후 크기 임계값($T_r$)이나 가로-세로 비율을 기준으로 부적절한 영역을 제거하고, 공간적으로 매우 가까운($T_c$ 기준) 영역들을 병합하여 최종 후보 영역(Candidate Object Regions, $\text{COR}_i$)을 생성한다.

### 2. 객체 추적 및 상태 결정

각 프레임 $t$에서 이전 프레임의 KCF 추적기 출력($TO_j^t$)와 현재의 $\text{COR}_i^t$ 사이의 중첩도를 계산하여 객체의 상태를 결정한다. 중첩도는 다음과 같은 IoU(Intersection over Union) 수식을 사용한다.

$$\text{Overlap}(x, y) = \frac{x \cap y}{x \cup y}$$

상태는 다음 네 가지로 분류된다.

- **Tracked (추적 중):** $\text{COR}_i^t$가 단 하나의 $TO_j^t$와만 겹칠 때이다. 기본적으로 배경 제거 결과인 $\text{COR}_i^t$를 사용하여 KCF의 스케일을 업데이트한다. 단, 배경 제거 결과가 너무 작아져 분절(Fragmentation)이 의심되는 경우($T_{ol}, T_{oh}$ 임계값 기준), KCF의 출력을 신뢰한다.
  $$CT_j^t = \begin{cases} CT_{j}^{t-1} \cup TO_j^t, & \text{if } T_{ol} \le \frac{A(TO_j^t)}{A(\text{COR}_i^t)} \le T_{oh} \\ CT_{j}^{t-1} \cup \text{COR}_i^t, & \text{otherwise} \end{cases}$$
  ($A(\cdot)$는 바운딩 박스의 면적을 의미한다.)

- **Occluded (폐색):** 하나의 $\text{COR}_i^t$에 여러 개의 $TO_j^t$가 겹칠 때이다. 이때는 배경 제거가 객체를 구분하지 못하는 상태이므로, KCF 추적기의 출력을 그대로 사용하여 개별 객체를 추적한다.
  $$CT_j^t = CT_{j}^{t-1} \cup TO_j^t$$
  만약 두 KCF 추적기가 동일 객체를 중복 추적하는 경우(Redundant trackers), 두 추적기 면적의 합이 $\text{COR}_i^t$의 면적보다 크면 더 최근에 생성된 추적기를 삭제한다.
  $$\text{if } A(TO_m^t) + A(TO_n^t) > A(\text{COR}_i^t), \text{Delete } KCF_n^t$$

- **New Object (신규 객체):** $\text{COR}_i^t$가 어떤 $TO_j^t$와도 겹치지 않을 때이다. 새로운 KCF 추적기를 생성하여 할당한다. 이때, 폐색 후 추적기가 표류(Drift)하여 발생한 문제인지 확인하기 위해 객체 그룹 관리 기법을 사용하여 추적기를 재할당한다.

- **Invisible or Exited (소멸 또는 이탈):** $TO_j^t$가 어떤 $\text{COR}_i^t$와도 겹치지 않을 때이다. 8프레임 이상 지속될 경우 해당 객체가 화면 밖으로 나간 것으로 간주하고 삭제한다.

## 📊 Results

### 실험 설정

- **데이터셋:** Urban Tracker 데이터셋의 4가지 영상(Sherbrooke, Rouen, St-Marc, Rene-Levesque)을 사용하였다.
- **비교 대상:** Urban Tracker (UT), Traffic Intelligence (TI), Mendes et al.의 추적기.
- **평가 지표:** CLEAR MOT 지표인 MOTA(Multiple Object Tracking Accuracy)와 MOTP(Multiple Object Tracking Precision)를 사용하였다. (MOTA는 높을수록, MOTP는 낮을수록 우수함)

### 정량적 결과 분석

- **MOTA 성능:** 제안 방법(MKCF)은 UT에 근접하는 경쟁력 있는 성능을 보였다. 특히 Rene-Levesque 영상의 자동차 및 자전거 추적에서는 가장 높은 MOTA를 기록하였다.
- **객체별 특성:** 보행자 추적의 경우 $\text{COR}_i^t$의 크기 임계값($T_r$) 설정에 따라 성능 변화가 컸는데, 너무 낮추면 환경 노이즈가 증가하고 너무 높이면 작은 보행자를 놓치는 Trade-off가 존재함을 확인하였다.
- **연산 속도:** Intel Core i5 CPU 환경에서 영상에 따라 6.2 ~ 18.6 FPS의 처리 속도를 보였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 매우 단순한 데이터 연관 기법을 사용했음에도 불구하고, KCF라는 강력한 단일 객체 추적기를 MOT 프레임워크에 통합함으로써 SOTA(State-of-the-art) 수준의 성능을 낼 수 있음을 입증하였다. 특히 폐색 상황에서 객체를 개별적으로 유지하는 방식이 기존의 merge-split 방식보다 효율적일 수 있음을 보여주었다.

### 한계 및 비판적 해석

- **정지 객체 문제:** 배경 제거 방식의 근본적인 한계로 인해, 객체가 오래 정지해 있으면 배경 모델에 통합된다. 이 객체가 다시 움직이기 시작할 때 여러 개의 새로운 객체로 인식되거나 분절되는 문제가 발생하며, 이는 실험 결과에서 자동차 추적 성능이 낮게 나타난 주요 원인으로 분석된다.
- **파라미터 의존성:** 영상마다 $T_r, T_c$ 등의 임계값을 다르게 설정해야 하는 점은 일반화 성능 측면에서 아쉬운 부분이다.
- **데이터 연관의 단순함:** 저자는 단순한 연관 기법이 효과적이었다고 주장하지만, 정지 후 재출발하는 객체의 분절 문제를 해결하기 위해서는 더 정교한 데이터 연관 알고리즘이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 도시 교통 환경에서 **배경 제거(검출 및 스케일 제공)와 KCF(폐색 시 개별 추적 및 데이터 연관)**를 결합한 MOT 시스템을 제안한다. 복잡한 알고리즘 없이도 두 방식의 상호 보완적 특성을 이용하여 폐색 문제를 효과적으로 해결하였으며, Urban Tracker 데이터셋에서 기존의 복잡한 추적기들과 경쟁 가능한 성능을 입증하였다. 이 연구는 강력한 단일 객체 추적기가 MOT 시스템의 전체적인 강건성을 높이는 데 핵심적인 역할을 할 수 있음을 시사한다.
