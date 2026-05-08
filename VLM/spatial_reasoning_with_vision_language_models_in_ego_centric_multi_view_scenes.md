# Spatial Reasoning with Vision-Language Models in Ego-Centric Multi-View Scenes

Mohsen Gholami, Ahmad Rezaei, Zhou Weimin, Yong Zhang, Mohammad Akbari (2025)

## 🧩 Problem to Solve

본 논문은 Vision-Language Models(VLMs)가 3D 공간 관계를 이해하는 능력이 매우 부족하다는 점을 해결하고자 한다. 기존의 공간 추론(spatial reasoning) 관련 데이터셋과 벤치마크는 주로 단일 이미지나 정적인 실내 환경에서 촬영된 비디오에 집중되어 있었다. 그러나 자율주행 자동차나 로봇과 같은 실제 Embodied AI 에이전트는 전방, 후방, 측면을 동시에 포착하는 ego-centric multi-view 관측치에 의존하여 동작한다.

이러한 multi-view 입력은 단순한 시각적 정보의 집합이 아니라, 에이전트의 기준 좌표계에 따른 명시적인 공간적 의미(예: '왼쪽'과 '오른쪽'의 고정된 방향성)를 내포하고 있다. 따라서 본 연구의 목표는 실제 야외 환경의 ego-centric multi-view 데이터를 활용하여 VLMs의 3D 공간 추론 능력을 정밀하게 평가할 수 있는 벤치마크를 구축하고, 이를 향상시키기 위한 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 크게 두 가지이다. 첫째, ego-centric multi-view 설정에서 VLMs의 3D 공간 이해 능력을 평가하는 최초의 벤치마크인 **Ego3D-Bench**를 제안한다. 이 벤치마크는 실제 야외 동적 환경을 반영하여 구축되었으며, 인간 수준의 공간 이해 능력과 VLM 간의 상당한 성능 격차를 드러낸다.

둘째, VLMs의 공간 추론 능력을 강화하기 위한 post-training 프레임워크인 **Ego3D-VLM**을 제안한다. 이 방법의 중심 아이디어는 복잡한 3D 포인트 클라우드나 BEV(Bird-Eye-View) 이미지 대신, 추론에 필요한 핵심 객체들의 3D 좌표만을 포함하는 **텍스트 기반의 인지 지도(textual cognitive map)**를 생성하여 VLM에 제공하는 것이다. 이를 통해 연산 효율성을 높이면서도 모델이 일관된 세계 모델(world model)을 구축하도록 돕는다.

## 📎 Related Works

기존의 공간 벤치마크인 VSI-Bench, CA-VQA, Q-Spatial-Bench 등은 단일 뷰나 정적인 실내 환경에 치중되어 있어, 동적인 야외 환경에서 동작하는 Embodied AI의 지각 경험을 충분히 반영하지 못한다. All-Angle Bench가 multi-view 설정을 도입했으나, 이는 주로 감시 카메라와 같은 고정된 다각도 뷰를 다루며 에이전트 중심의 ego-centric 관점과는 차이가 있다.

3D spatial VLMs 분야에서는 포인트 클라우드를 입력으로 사용하거나 이를 재구성하는 방식(3D-LLM, SpatialLM 등)과 이미지 데이터를 직접 사용하는 방식(LLaVA-3D, SpatialVLM 등)으로 나뉜다. 포인트 클라우드 기반 방식은 풍부한 정보를 제공하지만, 동적 환경에서의 재구성이 어렵고 추론 시간이 매우 길다는 단점이 있다. 본 연구의 Ego3D-VLM은 이미지 기반 카테고리에 속하며, 기존 모델들이 주로 실내 정적 장면이나 정성적 관계에 집중한 것과 달리, 야외 동적 장면에서의 정량적인 공간 관계 추론을 가능하게 한다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

Ego3D-VLM은 입력된 multi-view 이미지와 쿼리를 바탕으로 텍스트 형태의 인지 지도를 생성하여 VLM에 입력하는 구조를 가진다. 전체 과정은 다음과 같은 단계로 진행된다.

1. **객체 검출 (REC):** Referring Expression Comprehension(REC) 모델을 사용하여 쿼리에 언급된 객체의 2D 바운딩 박스 좌표 $b^{(v)}_i$와 중심점 $u^{(v)}_i$를 추출한다.
2. **3D 좌표 변환 (Coordinate Transformation):** Metric depth estimator를 통해 각 뷰의 깊이 맵 $D^{(v)}$를 생성하고, 객체 중심점의 깊이 값 $d^{(v)}_i$를 추출한다. 이후 카메라 내부 파라미터 $K^{(v)}$를 이용하여 2D 점을 3D 카메라 좌표계 $p^{(v)}_{cam,i}$로 투영한다.
    $$p^{(v)}_{cam,i} = d^{(v)}_i \cdot (K^{(v)})^{-1} [x_i, y_i, 1]^\top$$
    그 다음, 회전 행렬 $R^{(v)}$와 변환 벡터 $T^{(v)}$를 사용하여 이를 전방 카메라 기준의 글로벌 좌표계 $p^{(v)}_{global,i}$로 변환한다.
    $$p^{(v)}_{global,i} = \begin{bmatrix} R^{(v)} & T^{(v)} \\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} p^{(v)}_{cam,i} \\ 1 \end{bmatrix}$$
3. **관계형 스케일링 (Relational Scaling):** 실제 정답 깊이 값 없이도 물리적으로 타당한 스케일을 얻기 위해, 사람이나 차량과 같이 크기가 알려진 객체를 기준으로 스케일 인자 $s$를 계산하여 좌표를 조정한다.
    $$s = h_{cs} / h_{est}, \quad p^{(v)}_{scaled,i} = s \cdot p^{(v)}_{global,i}$$
    여기서 $h_{cs}$는 상식적인 표준 높이이며, $h_{est}$는 관측된 평균 높이다.
4. **인지 지도 생성 (Cognitive Map Generation):** 최종적으로 변환된 3D 좌표와 객체 이름들을 조합하여 텍스트 형태의 인지 지도 $C$를 생성하는 함수 $F_{cog}$를 정의한다.
    $$C = F_{cog}(\{p^{(v)}_{global,i}, c^{(v)}_i\}_{i,v})$$

### 추론 절차

최종 VLM $\mathcal{V}$는 생성된 텍스트 인지 지도 $C$, 원본 multi-view 이미지 세트 $I$, 그리고 자연어 쿼리 $q$를 동시에 입력받아 답변 $a$를 생성한다.
$$a = \mathcal{V}(C, I, q)$$
이때 인지 지도는 구조화된 공간적 근거를 제공하고, 이미지는 색상이나 외형과 같은 세부 시각적 단서를 제공함으로써 상호 보완적인 역할을 수행한다.

## 📊 Results

### 실험 설정

- **데이터셋:** nuScenes, Waymo Open Dataset, Argoverse 1의 validation set을 활용하여 8,600개 이상의 QA 쌍을 구축하였다.
- **평가 작업:** (1) 절대 거리 측정(Absolute Distance), (2) 상대 거리 측정(Relative Distance), (3) 위치 추론(Localization), (4) 모션 추론(Motion Reasoning), (5) 이동 시간 예측(Travel Time)의 5개 카테고리로 구성된다.
- **측정 지표:** 다지선다형 질문은 정확도(Accuracy), 절대 거리 예측은 RMSE(Root Mean Squared Error)를 사용한다.
- **비교 모델:** GPT-4o, Gemini-1.5-Pro 등 closed-source 모델과 InternVL3, Qwen2.5-VL 등 open-source 모델을 포함한 16종의 SOTA VLMs를 평가하였다.

### 주요 결과

- **VLM의 한계:** 대부분의 VLM은 인간 수준의 성능에 크게 못 미쳤으며, 특히 이동 시간 예측, 위치 추론, 객체 중심 절대 거리 측정에서 취약함을 보였다. 이는 VLM이 multi-view 이미지로부터 일관된 세계 모델을 구축하는 능력이 부족함을 시사한다.
- **Ego3D-VLM의 성능 향상:** Ego3D-VLM을 적용했을 때, 다지선다형 QA에서는 평균 12%의 정확도 향상이 있었으며, 절대 거리 예측 RMSE는 평균 56% 감소하는 괄목할 만한 성과를 거두었다.
- **인간 성능과의 비교:** 특히 객체 중심의 절대 거리 측정 작업에서는 Ego3D-VLM이 탑재된 모델이 인간의 성능을 능가하기도 했는데, 이는 인간이 명시적인 3D 정보 없이 거리를 추정할 때 발생하는 오차가 크기 때문으로 분석된다.
- **범용성 확인:** All-Angle Bench 및 VSI-Bench와 같은 다른 multi-view 설정에서도 baseline 대비 성능 향상을 보여, 제안 방법이 다양한 환경에서 적응 가능함을 입증하였다.

## 🧠 Insights & Discussion

본 연구는 VLMs가 단순히 이미지를 보는 것을 넘어, 여러 뷰의 정보를 통합하여 내부적인 '공간 지도'를 그려내야만 고차원적인 공간 추론이 가능하다는 점을 시사한다. 포인트 클라우드나 BEV 이미지 같은 고차원 표현 방식은 정보량은 많으나 연산 비용이 너무 크고 동적 환경에서 재구성이 어렵다. 반면, 본 논문이 제안한 **텍스트 기반 인지 지도**는 추론에 필요한 핵심 정보만을 압축적으로 전달함으로써 효율성과 정확성이라는 두 마리 토끼를 잡았다고 평가할 수 있다.

다만, 본 방법론은 기반이 되는 VLM의 기본 추론 능력에 의존하므로, 추론 능력이 매우 낮은 소형 모델(예: VILA1.5-8B)에서는 성능 향상 폭이 적다는 한계가 있다. 또한, 사용된 REC 모델의 오검출(false positive)로 인해 인지 지도에 불필요한 정보가 포함될 수 있으며, 야외 환경에서의 metric depth estimation의 정확도 한계로 인해 관계형 스케일링(Relational Scaling)과 같은 근사치 방법을 사용해야 했다는 점이 향후 개선 과제로 남는다.

## 📌 TL;DR

본 논문은 야외 ego-centric multi-view 환경에서 VLMs의 3D 공간 추론 능력을 평가하는 최초의 벤치마크인 **Ego3D-Bench**와, 이를 보완하기 위해 텍스트 기반의 3D 좌표 지도를 제공하는 **Ego3D-VLM** 프레임워크를 제안한다. 실험 결과, 텍스트 인지 지도를 통해 VLM의 공간 이해 능력을 획기적으로 향상시켰으며(RMSE 56% 개선), 이는 향후 자율주행 로봇이나 스마트 모빌리티와 같은 Embodied AI 에이전트의 공간 인지 능력을 높이는 데 핵심적인 역할을 할 가능성이 높다.
