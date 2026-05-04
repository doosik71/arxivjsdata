# Instance Segmentation GNNs for One-Shot Conformal Tracking at the LHC

Savannah Thais, Gage DeZoort (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 거대 강입자 가속기(Large Hadron Collider, LHC)와 같은 입자 가속기 환경에서 발생하는 입자 궤적 추적(Particle Tracking) 문제이다. 입자 검출기에서 측정된 수많은 히트(hit)들의 포인트 클라우드(point cloud)로부터 어떤 히트들이 하나의 입자 궤적에 속하는지를 식별하고, 해당 궤적의 물리적 파라미터를 추출하는 것이 핵심이다.

이 문제는 컴퓨터 비전의 3D 인스턴스 세그멘테이션(3D Instance Segmentation)으로 개념화할 수 있다. 특히 HL-LHC(High-Luminosity LHC)와 같이 입자 충돌 횟수가 매우 많은 고밀도 환경에서는 데이터의 양이 방대해지므로, 연산 효율적이면서도 정확하게 궤적을 찾고 파라미터를 추출하는 알고리즘이 필수적이다. 따라서 본 논문의 목표는 Graph Neural Networks(GNNs)와 컨포멀 기하학(Conformal Geometry)을 결합하여, 단 한 번의 패스(single shot)만으로 궤적 식별과 파라미터 추출을 동시에 수행하는 모델을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 입자 궤적 추적 문제를 단순한 엣지 분류(edge classification)가 아닌 **인스턴스 세그멘테이션** 관점에서 접근하고, 이를 위해 **컨포멀 매핑(Conformal Mapping)**을 도입한 것이다.

전통적인 방식은 GNN을 통해 히트 간의 연결 확률(edge weight)을 예측한 뒤, 별도의 후처리 알고리즘을 통해 궤적을 생성하고 파라미터를 추출했다. 반면, 본 논문은 다음과 같은 설계를 통해 'One-shot' 추적을 가능하게 한다.

1. **컨포멀 공간으로의 변환**: 원형의 입자 궤적을 직선 또는 포물선 형태로 변환하여 GNN이 궤적의 특징을 더 쉽게 학습하고 파라미터를 직접 추출할 수 있도록 한다.
2. **Instance Segmentation GNN 아키텍처**: 포인트 클라우드에서 직접 궤적 인스턴스를 탐지하고, 각 인스턴스의 바운딩 박스(bounding box)와 물리적 파라미터를 동시에 예측하는 파이프라인을 구축하였다.

## 📎 Related Works

기존의 포인트 클라우드 학습 방식은 데이터를 정규 그리드(regular grid)로 매핑한 뒤 CNN을 적용하는 방식이었으나, 이는 데이터의 희소성(sparsity)으로 인해 정보 손실이 발생하는 문제가 있었다. 최근에는 순서가 없는 포인트 집합을 직접 처리할 수 있는 GNN 기반의 3D 객체 탐지 및 세그멘테이션 연구들이 성과를 보이고 있다.

입자 물리학 분야에서는 GNN을 이용해 엣지 분류(edge classification)를 수행하는 방식이 제안된 바 있다. 이 방식은 메시지 패싱 GNN이 엣지의 가중치를 예측하면, 이후 secondary algorithm이 이를 바탕으로 궤적을 구성한다. 그러나 본 논문은 이러한 다단계 과정 대신, GNN이 직접 전체 궤적을 탐지하고 파라미터를 추출하는 단일 단계(single shot) 방식을 제안함으로써 기존 접근 방식과 차별점을 둔다.

## 🛠️ Methodology

### 1. 그래프 구축 (Graph Construction)

입자 히트들을 정점으로, 동일 입자에 의해 생성된 히트 간의 연결을 엣지로 하는 그래프 $G = (P, E)$를 생성한다.

- **정점(Vertex)**: 히트의 좌표 $(\eta, \phi, s)$와 상태 값 $s_i$ (z 좌표 및 레이어 번호)를 포함한다.
- **엣지(Edge)**: $\eta-\phi$ 공간에서 DBScan 클러스터링 알고리즘을 사용하여 유사한 방향으로 이동하는 히트들을 그룹화하여 연결한다.

### 2. Conformal GNN 아키텍처

본 모델은 Point-GNN에서 영감을 얻었으며, 다음의 세 가지 주요 구성 요소로 이루어져 있다.

**A. Auto-registration 기반 GNN**
메시지 패싱 과정을 통해 정점의 상태 값을 업데이트한다. 특히, 주변 이웃 간의 변환 분산(translation variance)을 줄이기 위해 **Auto-registration** 메커니즘을 도입하였다.
이전 반복 단계의 상태 값 $s_i^t$를 통해 정렬 오프셋 $\Delta x_i^t = h_t(s_i^t)$를 예측하며, 업데이트 식은 다음과 같다.
$$s_{t+1}^i = g_t(\rho(\{f(x_j - x_i + \Delta x_i^t, s_j^t)\}), s_i^t)$$
여기서 $f, g, h$는 MLP로 구현되며, $\rho$는 Max pooling을 사용한다.

**B. 바운딩 박스 지역화 (Bounding-box Localization)**
GNN의 출력값을 사용하여 각 정점을 '궤적 히트' 또는 '노이즈 히트'로 분류하고, 궤적 히트에 대해 $\eta-\phi$ 공간에서의 타원형 바운딩 박스 $B = (\eta_c, \phi_c, a, b, \theta)$를 예측한다. 여기서 $(\eta_c, \phi_c)$는 중심, $(a, b)$는 장축과 단축의 길이, $\theta$는 회전 각도이다.

**C. 궤적 파라미터 추출 (Track Parameter Extraction)**
최종적으로 분류 및 지역화 단계를 통해 형성된 클러스터를 컨포멀 공간(conformal space)으로 변환한 뒤, $\text{MLP}_t$를 통해 횡단면 운동량 $p_T$와 횡단 충격 파라미터 $\epsilon_T$를 직접 예측한다.

### 3. 손실 함수 (Loss Functions)

모델은 세 가지 손실 함수의 가중 합 $\mathcal{l}_{total} = \alpha l_c + \beta l_{loc} + \gamma l_t$를 최소화하도록 학습된다.

- **분류 손실 ($l_c$)**: 히트의 궤적 소속 여부를 판단하는 Binary Cross Entropy(BCE) 손실을 사용한다.
  $$l_c = -\frac{1}{n_{hits}} \sum_{i=1}^{n_{hits}} y_i \log y_i + (1-y_i) \log(1-y_i)$$
- **지역화 손실 ($l_{loc}$)**: 예측된 타원 바운딩 박스와 실제 박스 간의 차이를 Huber loss로 계산한다.
  $$l_{loc} = \frac{1}{n_{hits}} \sum_{i=1}^{n_{hits}} \mathbb{1}(v_i \in \{\text{trackhits}\}) l_{huber}(\delta - \delta_{gt})$$
- **추적 손실 ($l_t$)**: 예측된 $p_T$ 및 $\epsilon_T$와 실제 값 사이의 평균 제곱 오차(MSE)를 사용한다.
  $$l_t = \frac{1}{n_{clusters}} \sum_{i=1}^{n_{clusters}} \left( \frac{p_{T,i} - p_{T,i}^c}{p_T^c} \right)^2 + \left( \frac{\epsilon_{T,i} - \epsilon_{T,i}^c}{\epsilon_T^c} \right)^2$$

## 📊 Results

본 연구는 TrackML 데이터셋의 픽셀 검출기 데이터를 사용하여 모델을 평가하였다. 실험 환경으로는 PyTorch Geometric 라이브러리를 사용하였으며, $T=4$회의 GNN 반복과 Adam 옵티마이저(learning rate $10^{-6}$)를 적용하여 30 epoch 동안 학습을 진행하였다.

**정성적 결과:**

- 예측된 타원 바운딩 박스가 각 궤적을 효과적으로 지역화(localize)함을 확인하였다.
- $\eta-\phi$ 공간에서 궤적이 길게 늘어진 형태를 띠기 때문에, 동일 궤적 내의 히트들은 타원이 많이 겹치지만, 서로 다른 궤적 간에는 겹침이 적어 효과적인 분리가 가능함을 보였다.
- 다만, 실제 데이터에서 $\phi$ 좌표의 분포가 균일하지 않아 예측된 타원의 방향성에 편향(bias)이 발생하는 현상이 관찰되었다.

## 🧠 Insights & Discussion

**강점 및 가치:**
본 논문은 입자 추적 문제를 인스턴스 세그멘테이션으로 재정의함으로써, 궤적 탐지와 파라미터 추출을 단일 단계로 통합했다는 점에서 큰 의의가 있다. 특히 $\eta-\phi$ 공간과 컨포멀 공간의 물리적 대칭성을 활용하여 GNN의 학습 난이도를 낮추고 연산 효율성을 높였다.

**한계 및 미해결 과제:**

1. **중복 예측 문제**: 하나의 궤적에 대해 여러 개의 바운딩 박스가 예측되는 문제가 있으며, 이를 해결하기 위해 IoU(Intersection-Over-The-Union) 임계값을 이용한 Non-Maximum Suppression(NMS) 알고리즘의 도입이 필요하다.
2. **하이퍼파라미터 최적화**: $\text{MLP}_t$를 포함한 각 네트워크 층의 크기와 하이퍼파라미터에 대한 체계적인 분석이 아직 부족한 상태이다.
3. **공간 변환의 완전한 통합**: 현재는 지역화 단계가 $\eta-\phi$ 공간에서 이루어지지만, 이를 컨포멀 공간에서 직접 예측하도록 변경한다면 모델의 효율성을 더욱 높일 수 있을 것으로 보인다.

## 📌 TL;DR

본 논문은 LHC의 입자 궤적 추적 문제를 **3D 인스턴스 세그멘테이션**으로 접근하여, **GNN과 컨포멀 매핑**을 통해 궤적 탐지와 파라미터 추출을 한 번에 수행하는 'One-shot' 아키텍처를 제안하였다. 이 방식은 기존의 엣지 분류 방식보다 파이프라인을 단순화하며, 물리적 대칭성을 이용해 효율적인 추적이 가능함을 보였다. 향후 NMS 도입 및 컨포멀 공간으로의 완전한 통합을 통해 HL-LHC와 같은 고밀도 데이터 환경에서 핵심적인 추적 소프트웨어로 활용될 가능성이 높다.
