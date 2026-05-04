# Graph Attention Tracking

Dongyan Guo, Yanyan Shao, Ying Cui, Zhenhua Wang, Liyan Zhang, Chunhua Shen (2020)

## 🧩 Problem to Solve

본 논문은 기존의 Siamese 네트워크 기반 트래커들이 가진 근본적인 한계점을 해결하고자 한다. 대부분의 Siamese 트래커는 타겟 브랜치(target branch)와 서치 브랜치(search branch) 간의 Convolutional feature cross-correlation을 통해 유사도를 학습한다. 그러나 이러한 방식은 다음과 같은 세 가지 주요 문제를 야기한다.

첫째, **고정된 커널 크기의 문제**이다. 템플릿 특징 영역의 크기가 미리 정해져 있기 때문에, 객체의 크기나 종횡비(aspect ratio)가 변화할 경우 불필요한 배경 정보가 포함되거나 중요한 전경 정보가 누락되어 정보 임베딩의 정확도가 떨어진다.

둘째, **전역 매칭(Global Matching)의 한계**이다. 타겟과 서치 영역을 전체적으로 매칭하는 방식은 객체의 세부 구조나 부분별(part-level) 정보를 무시한다. 이로 인해 객체의 회전, 포즈 변화, 심한 가려짐(occlusion)이 발생했을 때 강건하게 대응하지 못한다.

셋째, **과도한 정보 압축**이다. 전역 매칭 프로세스는 템플릿에서 서치 영역으로 전달되는 정보를 과도하게 압축하여, 정교한 위치 추정에 필요한 세부 정보의 손실을 초래한다.

따라서 본 논문의 목표는 객체의 부분적 관계를 학습하고 객체의 크기 및 종횡비 변화에 적응할 수 있는 새로운 정보 임베딩 메커니즘을 제안하여 일반 객체 추적(General Object Tracking)의 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Graph Attention Network (GAT)**를 도입하여 템플릿과 서치 영역 간의 **부분-대-부분(part-to-part) 대응 관계**를 구축하는 것이다.

1.  **Graph Attention Module (GAM) 제안**: 템플릿과 서치 영역의 각 특징 점을 노드로 하는 완전 이분 그래프(complete bipartite graph)를 구성하고, Attention 메커니즘을 통해 템플릿의 정보를 서치 특징으로 효율적으로 전파한다. 이를 통해 전역 매칭의 한계를 극복하고 부분별 불변 특징을 활용할 수 있게 한다.
2.  **Target-aware Area Selection 메커니즘**: 고정된 영역을 크롭하는 대신, 타겟의 실제 바운딩 박스에 기반하여 특징 영역을 선택함으로써 객체의 크기와 종횡비 변화에 유연하게 대응한다.
3.  **SiamGAT 프레임워크**: 위 두 가지 요소를 결합하여 단순하면서도 효과적인 추적 프레임워크를 구축하였으며, 여러 벤치마크 데이터셋에서 최신 트래커들보다 우수한 성능을 입증하였다.

## 📎 Related Works

기존의 Siamese 트래커들은 주로 특징 추출 네트워크의 최적화나 바운딩 박스 회귀를 위한 추적 헤드(tracking head) 설계에 집중해 왔다.

- **Cross-correlation 기반 방식**: SiamFC는 템플릿 특징을 커널로 사용하여 서치 영역과 합성곱 연산을 수행한다. 이후 DSiam, SA-Siam, RASNet 등이 이 구조를 개선하였으나, 여전히 전역 매칭의 한계를 가지고 있다.
- **RPN 기반 방식**: SiamRPN 및 SiamRPN++는 Region Proposal Network를 도입하여 성능을 높였으며, 특히 SiamRPN++의 Depth-wise cross-correlation (DW-Xcorr)는 파라미터 분포 불균형을 해결하여 안정적인 학습을 가능하게 했다.
- **Anchor-free 방식**: SiamFC++, SiamCAR, Ocean 등은 앵커 없이 픽셀 단위로 예측하는 방식을 도입하여 하이퍼파라미터 튜닝의 번거로움을 줄이고 성능을 향상시켰다.

본 연구는 이러한 기존 연구들이 공통적으로 사용하는 **고정 크기 템플릿**과 **전역 매칭(cross-correlation)** 방식이 정보 전파의 병목 현상을 일으킨다고 지적하며, 이를 Graph Attention 기반의 부분-대-부분 매칭으로 차별화한다.

## 🛠️ Methodology

### 전체 파이프라인
SiamGAT는 크게 세 가지 블록으로 구성된다: **Siamese 특징 추출 네트워크 $\rightarrow$ Graph Attention Module (GAM) $\rightarrow$ 분류-회귀 헤드(Classification-Regression Head)**. 특징 추출을 위해 GoogLeNet (Inception v3)을 백본으로 사용하며, 최종 위치 예측을 위해 SiamCAR의 헤드 구조를 채택하였다.

### Graph Attention Module (GAM)
GAM은 템플릿 특징 맵 $F_t$와 서치 특징 맵 $F_s$를 각각 노드 집합 $V_t$와 $V_s$로 간주하고, 이들 사이의 관계를 완전 이분 그래프 $G = (V_s \cup V_t, E)$로 모델링한다.

1.  **상관성 점수(Correlation Score) 계산**:
    서치 노드 $i \in V_s$와 템플릿 노드 $j \in V_t$ 사이의 유사도를 측정하기 위해 선형 변환 후 내적을 수행한다.
    $$e_{ij} = (W_s h_i^s)^T (W_t h_j^t)$$
    여기서 $h_i^s$와 $h_j^t$는 각 노드의 특징 벡터이며, $W_s$와 $W_t$는 학습 가능한 선형 변환 행렬이다.

2.  **Attention 가중치 산출**:
    Softmax 함수를 사용하여 노드 $i$가 템플릿의 각 부분 $j$에 얼마나 집중해야 하는지를 정규화한다.
    $$a_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in V_t} \exp(e_{ik})}$$

3.  **특징 집계(Aggregation)**:
    계산된 가중치를 바탕으로 템플릿의 정보를 집계하여 노드 $i$의 표현을 재구성한다.
    $$v_i = \sum_{j \in V_t} a_{ij} W_v h_j^t$$
    여기서 $W_v$는 또 다른 선형 변환 행렬이다.

4.  **최종 특징 융합**:
    집계된 특징 $v_i$와 원래의 서치 특징 $h_i^s$를 연결(concatenation)한 후 ReLU 활성화 함수를 통과시켜 최종 응답 맵(response map)을 생성한다.
    $$\hat{h}_i^s = \text{ReLU}(v_i \parallel (W_v h_i^s))$$

### Target-aware Area Selection
기존의 고정된 $m \times m$ 영역 크롭 방식 대신, 템플릿 패치 내의 실제 바운딩 박스 $B_t$를 특징 맵 $F_t$에 투영하여 관심 영역 $R_t$를 설정한다.
$$\hat{F}_t = [F_t(i, j, :)]_{(i,j) \in R_t}$$
이렇게 얻어진 $\hat{F}_t$는 객체의 실제 크기와 종횡비에 맞는 텐서가 되며, 이 영역의 픽셀들만이 그래프의 노드가 된다. 또한, 서로 다른 크기의 $\hat{F}_t$를 처리하기 위해 $R_t$ 외부 영역을 0으로 채우는 방식을 사용하여 scale-invariant한 연산이 가능하게 함으로써 Batch Normalization 적용 문제를 해결하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: UAV123, GOT-10k, OTB-100, LaSOT.
- **백본**: GoogLeNet (Inception v3), ImageNet 사전 학습 가중치 사용.
- **비교 대상**: SiamCAR, SiamRPN++, Ocean-online/offline, SiamFC++ 등.
- **평가 지표**: Success rate (SR), Precision, Average Overlap (AO).

### 주요 결과
1.  **Ablation Study (UAV123)**:
    - 백본을 AlexNet에서 GoogLeNet으로 변경 시 Success rate가 $59.2\% \rightarrow 64.6\%$로 크게 향상되었다.
    - Target-aware 선택 메커니즘을 적용했을 때, 고정 영역 방식보다 Success와 Precision에서 각각 $2.0\%$, $2.1\%$의 성능 이득을 얻었다.
    - GAM 방식은 가장 강력한 Cross-correlation 방식인 DW-Xcorr보다 성능이 우수함을 확인하였다.

2.  **벤치마크 성능**:
    - **GOT-10k**: AO, $SR_{0.5}$, $SR_{0.75}$ 모든 지표에서 최상위 성능을 기록하였다. 특히 온라인 업데이트를 수행하는 'Ocean'보다도 단순한 구조임에도 불구하고 더 높은 성능을 보였다.
    - **UAV123**: Precision과 Success plot 모두에서 다른 트래커들을 압도하였다.
    - **OTB-100**: Success score $71.0\%$를 달성하며 최고 성능을 기록했다. 특히 변형(DEF), 회전(OPR, IPR), 가려짐(OCC) 상황에서 강건함을 보였다.
    - **LaSOT**: 온라인 업데이트 모델인 Ocean-online 다음으로 높은 성능을 기록했으며, 오프라인 모델들 중에서는 최상위권에 위치하였다.

## 🧠 Insights & Discussion

### 강점
본 논문은 전역 매칭 방식에서 부분-대-부분 매칭 방식으로의 전환이 추적 성능에 결정적인 영향을 미침을 증명하였다. 특히 Graph Attention을 통해 템플릿의 정보를 유연하게 전파함으로써, 객체의 포즈 변화나 모양 변형이 심한 상황에서도 강건하게 추적할 수 있는 능력을 갖추게 되었다. 또한, Target-aware 영역 선택을 통해 배경 노이즈를 효과적으로 제거한 점이 성능 향상의 주요 요인으로 분석된다.

### 한계 및 논의사항
실험 결과, 빠른 움직임(Fast Motion), 시야 이탈(Out-of-view), 저해상도(Low Resolution) 상황에서는 baseline인 SiamCAR보다 성능이 낮게 나타나는 경향이 있다. 이는 GAM이 형태 변화에는 강하지만, 급격한 외관 변화나 해상도 저하로 인해 노드 간의 유사도 측정 자체가 어려워지는 상황에서는 취약할 수 있음을 시사한다. 또한, 현재 모델은 오프라인 특징 추출에 의존하므로, 향후 온라인 업데이트 모듈을 통합한다면 장기 추적(long-term tracking) 성능을 더욱 높일 수 있을 것으로 보인다.

## 📌 TL;DR

본 논문은 기존 Siamese 트래커의 고정 커널 크기와 전역 매칭 문제를 해결하기 위해 **Graph Attention Module (GAM)**과 **Target-aware 영역 선택** 메커니즘을 제안한 **SiamGAT**를 소개한다. 템플릿과 서치 영역을 그래프 노드로 구성하여 부분적 대응 관계를 학습함으로써, 객체의 크기 변화와 포즈 변형에 매우 강건한 추적 성능을 구현하였다. 이 연구는 단순한 구조만으로도 최신 SOTA 트래커들을 능가하는 성능을 보였으며, 향후 Graph 기반의 정보 임베딩 방식이 비전 추적 분야에서 중요한 역할을 할 가능성을 제시하였다.