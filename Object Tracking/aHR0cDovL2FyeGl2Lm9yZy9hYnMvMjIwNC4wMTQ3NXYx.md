# Unsupervised Learning of Accurate Siamese Tracking

Qiuhong Shen, Lei Qiao, Jinyang Guo, Peixia Li, Xin Li, Bo Li, Weitao Feng, Weihao Gan, Wei Wu, Wanli Ouyang

## 🧩 Problem to Solve

기존의 비지도 시각 객체 추적(unsupervised visual object tracking) 연구들은 주로 단일 프레임에서 템플릿-탐색 쌍(template-search pairs)을 구성하는 공간적 지도(spatial supervision)에 의존하여, 장시간에 걸친 큰 객체 변화에 대처하는 데 한계가 있었습니다. 특히 바운딩 박스(box-regression) 추정 브랜치에 대한 시간적 대응(temporal correspondence) 학습이 부족했습니다.

이러한 비지도 학습 방식의 순환 학습(cycle training) 과정에서 세 가지 주요 문제가 식별되었습니다:

1. **전방 전파(forward propagation) 중 템플릿 손실:** 초기 학습 단계에서 트래커가 중간 프레임에서 객체를 놓쳐 템플릿 커널에 대상 객체 특징이 포함되지 않아 학습 파이프라인이 중단될 수 있습니다.
2. **후방 전파(backward propagation) 중 불량 조건 페널티(ill-posed penalty) 및 기울기 전달 불가:** RoI-Align과 같은 선택 연산은 박스 좌표에 대해 미분 불가능하여, 중간 프레임의 추적 오류에 대한 기울기가 역전파되지 않아 페널티를 줄 수 없습니다. 또한 부정확한 추적 결과로 생성된 노이즈 템플릿을 사용하여 정확한 예측을 강제하면 학습이 불안정해집니다.
3. **노이즈 유사 라벨(pseudo labels) 문제:** 광학 흐름(optical flow) 모델로 생성된 초기 프레임의 유사 라벨은 노이즈를 포함할 수 있어 추적 성능을 저하시킬 수 있습니다.

## ✨ Key Contributions

이 논문은 위에서 언급된 문제들을 해결하기 위한 새로운 비지도 추적 프레임워크인 ULAST(Unsupervised Learning of Accurate Siamese Tracking)를 제안하며, 다음과 같은 핵심 기여를 합니다:

- 분류(classification) 및 회귀(regression) 브랜치 모두에서 시간적 대응을 학습할 수 있는 **새로운 비지도 학습 프레임워크 ULAST**를 제안합니다.
- 전방 전파 과정에서 신뢰할 수 있는 템플릿 커널을 생성하여 ULAST 프레임워크의 학습 중단을 방지하는 **일관성 전파 변환(Consistency Propagation Transformation, CPT)**을 제안합니다.
- 특징을 선택하고 후방 전파 과정에서 중간 프레임의 추적 오류에 암묵적으로 페널티를 부여하는 **미분 가능한 영역 마스크(Region Mask) 연산**을 제안합니다.
- 노이즈 유사 라벨의 부정적인 영향을 완화하기 위해 유사 라벨의 품질에 따라 동적인 가중치를 할당하는 **마스크 기반 손실 재가중치(Mask-guided Loss Re-weighting) 전략**을 제안합니다.

## 📎 Related Works

- **지도 시각 추적 (Supervised Visual Tracking):** SiamFC, SiamRPN++, ATOM, DiMP와 같은 딥러닝 기반 트래커들이 주류를 이루며 뛰어난 성능을 보이지만, 방대한 양의 레이블링된 비디오 데이터셋을 필요로 합니다.
- **비지도 시각 추적 (Unsupervised Visual Tracking):**
  - **공간적 자기 지도(Spatial Self-Supervision):** s2siamfc와 같이 단일 프레임 내에서 자기 지도 신호를 활용하지만, 장시간의 객체 변화 학습에 한계가 있습니다.
  - **시간적 자기 지도(Temporal Self-Supervision):** UDT, USOT, PUL 등은 비디오 내의 순환 일관성(cycle consistency)을 활용하여 트래커를 학습합니다. 하지만 이들은 주로 분류 능력에 집중하거나 바운딩 박스 회귀에 대한 시간적 대응 학습이 부족합니다. 본 연구는 이러한 시간적 자기 지도 신호를 활용하여 분류와 회귀 능력을 동시에 학습하는 데 중점을 둡니다.

## 🛠️ Methodology

ULAST는 Siamese 네트워크 기반의 영역 제안 네트워크(Region Proposal Network, RPN) 구조를 채택하며, 주로 세 가지 새로운 구성 요소로 순환 학습을 통해 비지도 방식으로 추적 성능을 향상시킵니다.

1. **사이클 학습 (Cycle Training):**

   - 초기 프레임의 유사 라벨이 주어진 상태에서, 팔린드롬(palindrome) 순서(예: $1 \to 2 \to 3 \to 2 \to 1$)로 프레임을 추적하여 첫 번째 프레임으로 되돌아옵니다.
   - 최종 추적 결과와 첫 번째 프레임의 유사 라벨 간의 불일치(inconsistency)를 활용하여 트래커를 최적화합니다.
   - 총 손실 함수는 $L_{total} = (1 - \lambda_{c})L_{l} + \lambda_{c}L_{c}$로 구성됩니다. 여기서 $L_{l}$은 단일 프레임에서 템플릿-탐색 쌍을 구성하는 기존 학습 손실이고, $L_{c}$는 사이클 학습 손실입니다.

2. **영역 마스크 연산 (Region Mask Operation):**

   - 기존 RoI-Align의 미분 불가능성 및 불량 조건 페널티 문제를 해결하기 위해 제안됩니다.
   - RPN의 분류($P_{cls}$) 및 회귀($P_{reg}$) 결과로부터 지역 단위 특징을 선택하는 미분 가능한 영역 마스크 $M_t \in \mathbb{R}^{H \times W}$를 생성합니다.
   - 예측된 각 바운딩 박스와 검색 영역 특징의 그리드 셀 간의 IoU(Intersection over Union) 비율을 $G_{(i,j)}^k$로 계산하며, 이는 좌표에 대해 자연스럽게 미분 가능합니다.
   - 모든 그리드 맵 $\{G_k\}$와 해당 신뢰도 $s_k$를 결합하여 최종 영역 마스크 $M_t = \sum_{k=1}^K \mathbb{1}(s_k, TH) \cdot s_k \tilde{G}_k$를 생성합니다. (여기서 $\mathbb{1}$은 지시 함수, $TH$는 임계값).

3. **일관성 전파 변환 (Consistency Propagation Transformation, CPT):**

   - 트래커가 대상 객체를 놓치지 않고 프레임 간의 일관성 전파를 보장하기 위해 도입됩니다.
   - 현재 검색 프레임 특징 $S_t$와 영역 마스크 $M_t$를 기반으로 새로운 템플릿 커널 $T_t$를 생성합니다.
   - $\tilde{S}_t = S_t \otimes M_t$ (요소별 곱셈)를 통해 노이즈 특징을 제거합니다.
   - **장기 (Long-term) 특징 $X^{L}_{t}$:** 초기 템플릿 특징 $T_1$을 쿼리(query)로, $\tilde{S}_t$를 키(key) 및 값(value)으로 사용하여 대상의 가장 신뢰할 수 있는 특징을 추출합니다.
   - **단기 (Short-term) 특징 $X^{S}_{t}$:** 이전 프레임에서 계산된 숨겨진 템플릿 $H_{t-1}$을 쿼리로 사용하여 가장 최근의 대상 특징을 검색합니다.
   - 최종 템플릿 $T_t = h_\theta(\text{concat}(X^{S}_{t}, X^{L}_{t}))$는 장기 및 단기 특징을 결합하여 생성됩니다.

4. **마스크 기반 손실 재가중치 (Mask-guided Loss Re-weighting):**

   - 노이즈 유사 라벨의 부정적인 영향을 줄이기 위해 샘플별 동적 가중치 $w_b$를 할당합니다.
   - RPN의 출력으로 생성된 영역 마스크 $\hat{M}_b$와 유사 라벨 기반의 마스크 $\bar{M}_b$를 사용하여 가중치 $w_b = \log_{\gamma}(\alpha - \frac{\sum_p \sum_q \mathbb{1}(\hat{M}^b_{p,q}, \beta)}{\sum_p \sum_q \mathbb{1}(\bar{M}^b_{p,q}, \beta)})$를 계산합니다. (여기서 $\gamma, \alpha, \beta$는 하이퍼파라미터).
   - 총 손실 $L = \frac{1}{B} \sum_b w_b(\lambda_1 L_{cls} + \lambda_2 L_{reg})$에 계산된 가중치를 적용합니다.

5. **온라인 추적 (Online Tracking):**
   - 오프라인 학습 후, ULAST는 SiamRPN처럼 고속으로 동작할 수 있습니다 (80 FPS).
   - 강건한 추적을 위해 과거 검색 특징과 영역 마스크가 저장된 메모리 큐(memory queue)를 유지하여 템플릿 커널을 업데이트합니다.
   - 최종 분류 맵 $R_{cls}$는 초기 템플릿 커널과 메모리 커널의 $R_{cls} = (1 - \lambda_{m})R^{L}_{cls} + \lambda_{m}R^{M}_{cls}$를 결합하여 생성됩니다.

## 📊 Results

- **VOT2016, VOT2018, OTB2015, TrackingNet, LaSOT** 등 5개 벤치마크 데이터셋에서 광범위한 실험을 통해 ULAST의 효과가 입증되었습니다.
- **모든 벤치마크에서 기존 비지도 추적 방법들을 크게 능가**하는 성능을 보였습니다. 특히 TrackingNet과 LaSOT와 같은 대규모 데이터셋에서는 **지도 학습 방식과 견줄 만한(on par) 성능**을 달성했습니다.
- **ULAST\*-on (온라인 업데이트 적용)**은 ULAST\*-off (오프라인 추적 모드)보다 전반적으로 더 나은 성능을 보여, 메모리 업데이트의 효과를 입증했습니다.
- Ablation Study를 통해 CPT 모듈, 영역 마스크, 마스크 기반 손실 재가중치 전략이 각각 성능 향상에 기여함을 확인했습니다. 특히 CPT에서 초기 템플릿 특징을 잔여 연결(residual connection)로 포함하지 않는 것이 더 나은 성능을 보였습니다.
- ImageNet 사전 학습이 성능 향상에 긍정적인 영향을 미친다는 것을 보여주었습니다.

## 🧠 Insights & Discussion

- ULAST는 비지도 학습만으로도 지도 학습 기반의 최신 추적기와 견줄 만한 성능을 달성하여, 레이블링 비용이 많이 드는 문제를 해결할 수 있는 잠재력을 보여주었습니다.
- CPT, 영역 마스크, 마스크 기반 손실 재가중치와 같은 제안된 구성 요소들이 순환 학습의 주요 문제점(템플릿 손실, 미분 불가, 노이즈 라벨)을 효과적으로 해결했음을 확인했습니다. 특히, 영역 마스크의 임계값을 0으로 설정하여 모든 예측 박스를 고려하는 것이 더 많은 학습 샘플을 축적하고 트래커가 전경과 배경을 더 잘 구분하도록 학습시키는 데 유리하다는 통찰을 제공했습니다.
- CPT 모듈의 장기 및 단기 쿼리(query)는 각각 객체의 불변 영역과 가변 영역에 더 높은 응답을 보여, 상호 보완적으로 신뢰성 있는 템플릿 특징을 생성합니다.
- **한계점:** 여전히 초기 프레임의 유사 라벨 생성에 비지도 광학 흐름 모델에 의존합니다. 더 나은 초기화 방법과 더 나은 트래커 간의 "닭이 먼저냐 달걀이 먼저냐" 하는 난제(chicken-egg conundrum)가 남아있어, 초기화 방법과 비지도 추적을 하나의 End-to-End 학습 가능한 파이프라인으로 연결하는 것이 향후 과제입니다.

## 📌 TL;DR

- **문제:** 기존 비지도 시각 추적은 긴 시간 동안의 객체 변화에 취약하며, 바운딩 박스 추정을 위한 시간적 대응 학습이 부족했습니다. 특히 순환 학습(cycle training) 시 템플릿 손실, 기울기 전달 문제, 노이즈 라벨 문제가 있었습니다.
- **방법:** ULAST는 순환 학습을 통해 비지도 시각 추적을 수행합니다. 이를 위해 (1) **일관성 전파 변환 (CPT)**을 도입하여 신뢰할 수 있는 템플릿 커널을 생성하고, (2) **미분 가능한 영역 마스크 (Region Mask)**를 사용하여 중간 프레임의 추적 오류에 암묵적으로 페널티를 부여하며, (3) **마스크 기반 손실 재가중치 (Mask-guided Loss Re-weighting)** 전략으로 노이즈 라벨의 영향을 줄입니다.
- **발견:** ULAST는 기존 비지도 추적 방법들을 크게 능가하며, TrackingNet 및 LaSOT와 같은 대규모 데이터셋에서 지도 학습 방식과 비슷한 성능을 달성하여, 레이블링되지 않은 비디오에서 객체 추적 능력을 효과적으로 학습할 수 있음을 입증했습니다.
