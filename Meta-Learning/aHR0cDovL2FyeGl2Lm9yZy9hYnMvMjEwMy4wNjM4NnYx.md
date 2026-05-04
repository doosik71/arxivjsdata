# Improving Context-Based Meta-Reinforcement Learning with Self-Supervised Trajectory Contrastive Learning

Bernie Wang, Simon Xu, Kurt Keutzer, Yang Gao, Bichen Wu (2021)

## 🧩 Problem to Solve

본 논문은 Meta-Reinforcement Learning(Meta-RL)의 심각한 샘플 효율성 저하 문제를 해결하고자 한다. Meta-RL은 새로운 작업에 빠르게 적응하는 것을 목표로 하지만, 실제로는 단일 작업 RL(single-task RL)보다 수십 배 더 많은 샘플을 필요로 한다.

이러한 비효율성의 주요 원인은 Meta-training 단계에서 에이전트가 다양한 작업 분포를 처리해야 하며, 특히 Context Encoder와 같은 추가적인 구성 요소를 함께 학습시켜야 하기 때문이다. 기존의 Context-based Meta-RL 방법론들은 Context Encoder와 Policy를 단순히 end-to-end 방식으로 학습시키며, 오직 희소한(sparse) 작업 보상(task reward)에만 의존하여 감독 신호를 얻는다. 이로 인해 감독 신호가 약해 학습이 불안정해지고 성능이 저하되는 문제가 발생한다.

따라서 본 연구의 목표는 Context Encoder에 강력한 감독 신호를 직접 제공할 수 있는 자기지도 학습(self-supervised learning) 태스크를 도입하여 Meta-training의 효율성과 성능을 개선하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Trajectory Contrastive Learning (TCL)**이라는 새로운 보조 학습 태스크를 제안하는 것이다. 

TCL의 중심 직관은 "동일한 궤적(trajectory)에서 추출된 서로 다른 시점의 전이 윈도우(transition windows)들은 동일한 작업 특성을 공유할 것"이라는 점이다. 따라서 두 윈도우가 같은 궤적에서 왔는지 여부를 예측하는 대조 학습(contrastive learning)을 통해, Context Encoder가 보상 신호 없이도 작업의 고유한 특성(역학 및 보상 구조 등)을 반영하는 의미 있는 표현(representation)을 학습하도록 유도한다. 

이 방식은 추가적인 라벨링이 필요 없으며, 기존의 Context-based Meta-RL 알고리즘에 쉽게 결합할 수 있는 plug-and-play 구조를 가진다.

## 📎 Related Works

### 관련 연구 및 한계
1.  **Meta-Learning**: Gradient-based 방법(MAML 등)은 파라미터 초기화를 학습하여 빠른 최적화를 꾀하며, Context-based 방법(PEARL 등)은 과거의 경험을 context로 활용하여 적응한다. 하지만 Context-based 방법들은 Context Encoder 학습 시 보상 신호의 간접적인 감독에만 의존한다는 한계가 있다.
2.  **Auxiliary Tasks in RL**: 미래 상태 예측이나 재구성(reconstruction) 등의 보조 태스크가 샘플 효율성을 높이는 데 사용되어 왔다. 대조 학습 또한 시각적 표현 학습 등에 적용되었으나, Meta-RL 영역에서 작업 표현(task representation)을 학습하려는 시도는 부족했다.
3.  **Contrastive Learning**: SimCLR, MoCo 등 이미지 인식 분야에서 큰 성공을 거두었으며, RL에서는 CURL 등이 시각적 입력의 표현 학습에 이를 적용하였다.

### 기존 접근 방식과의 차별점
기존의 RL 보조 태스크들이 주로 상태 관측치(state observation)의 표현력을 높이는 데 집중한 반면, TCL은 Meta-RL의 계층적 구조를 활용하여 **작업(task) 자체의 표현력**을 높이는 데 집중한다. 또한, 확률적 컨텍스트 인코더(probabilistic context encoder)를 사용하므로, 단순한 벡터 비교가 아닌 확률 분포 간의 거리를 측정하는 방식을 채택하였다.

## 🛠️ Methodology

### 전체 시스템 구조
TCL은 기존의 Context-based Meta-RL 프레임워크 위에 추가되는 보조 손실 함수 형태로 작동한다. 전체 구조는 크게 **Context Encoder**와 **Conditional Policy**로 나뉜다. Context Encoder는 탐색 궤적을 입력받아 작업 임베딩 $z$를 생성하고, Conditional Policy는 이 $z$에 조건화되어 행동을 결정한다. TCL은 이 과정에서 Context Encoder가 더 정교한 $z$를 생성하도록 돕는다.

### Trajectory Contrastive Learning (TCL) 상세
TCL은 리플레이 버퍼에서 샘플링된 궤적 윈도우들을 사용하여 수행된다.

1.  **데이터 구성**: 리플레이 버퍼에서 궤적 $i$로부터 길이 $W$인 두 개의 윈도우 $w_{i,t}$와 $w_{i,t'}$를 추출하면 이를 **Positive pair**로 정의한다. 서로 다른 궤적 $i, j$에서 추출된 윈도우 $w_{i,t}$와 $w_{j,t'}$는 **Negative pair**가 된다.
2.  **Key-Query 아키텍처**: MoCo와 CURL의 구조를 따라 두 개의 인코더를 운용한다.
    -   **Query Encoder ($f_{\phi_q}$)**: 경사 하강법(gradient update)을 통해 직접 학습된다.
    -   **Key Encoder ($f_{\phi_k}$)**: Query Encoder의 파라미터를 지수 이동 평균(Exponential Moving Average, EMA) 방식으로 업데이트하여 안정적인 타겟을 제공한다.
3.  **유사도 측정 (Similarity Metric)**:
    Context 임베딩이 가우시안 분포 $N(\mu, \sigma)$로 모델링되므로, 단순한 코사인 유사도 대신 **Negative Wasserstein Distance**를 사용하여 분포 간의 유사도를 측정한다.
    $$\text{sim}(q,k) = -\left( \|f^\mu_{\phi_q}(w_{i,t}) - f^\mu_{\phi_k}(w_{i',t'})\|_2^2 + \|f^\sigma_{\phi_q}(w_{i,t}) - f^\sigma_{\phi_k}(w_{i',t'})\|_2^2 \right)$$
    여기서 $\mu$는 평균, $\sigma$는 표준편차를 의미하며, 두 분포의 평균과 표준편차 차이를 각각 계산하여 합산한다.
4.  **손실 함수 (Loss Function)**:
    InfoNCE 손실 함수를 사용하여, 쿼리와 동일 궤적의 키(positive) 사이의 유사도는 높이고, 다른 궤적의 키(negative)들과의 유사도는 낮춘다.
    $$L_{TCL} = -\frac{1}{N} \sum_{i=0}^{N-1} \log \frac{\exp(\text{sim}(q_i, k_i))}{\sum_{j=0}^{N-1} \exp(\text{sim}(q_i, k_j))}$$

### 학습 절차
Meta-training 과정에서 다음과 같은 단계가 추가된다.
-   리플레이 버퍼에서 궤적 윈도우를 샘플링하여 쿼리와 키 세트를 생성한다.
-   기존의 Meta-RL 손실 함수($L_{A_\theta}$)와 TCL 보조 손실 함수($L_{TCL}$)를 합산하여 $\phi_q$를 업데이트한다.
-   업데이트된 $\phi_q$를 이용하여 $\phi_k$를 EMA 방식으로 갱신한다.
-   테스트 단계에서는 TCL 구성 요소가 제거되며, 학습된 Context Encoder와 Policy만을 사용하여 빠르게 적응한다.

## 📊 Results

### 실험 설정
-   **데이터셋/벤치마크**: MuJoCo Controls (6개 환경) 및 Meta-World ML1 (50개 환경).
-   **비교 대상 (Baselines)**: PEARL (TCL을 적용하지 않은 기본 모델), MAML, RL2, MQL.
-   **Oracle**: 정답 작업 벡터(ground-truth task vector)를 직접 입력으로 제공한 상한선(upper bound).
-   **지표**: Meta-testing 단계에서의 평균 리턴(Average Return).

### 주요 결과
1.  **MuJoCo Controls**: 6개 환경 중 5개에서 PEARL보다 우수하거나 대등한 성능을 보였으며, PEARL 대비 평균 1.13배, 중앙값 1.14배의 성능 향상을 기록하였다.
2.  **Meta-World ML1**: 50개 환경 중 44개에서 PEARL보다 우수하거나 대등한 성능을 보였다. 특히 평균 성능 향상이 4.3배, 중앙값이 1.4배에 달해 MuJoCo보다 훨씬 큰 개선 효과를 보였다.
3.  **샘플 효율성**: Meta-World의 25개 환경 실험에서 TCL-PEARL이 가장 많은 '승리(wins, 타 알고리즘 대비 최고 성능)' 횟수(7회)를 기록하여 샘플 효율성이 뛰어남을 입증하였다.
4.  **정성적 분석 (t-SNE)**: 학습된 컨텍스트 임베딩 공간을 시각화한 결과, TCL-PEARL은 PEARL에 비해 동일 작업 내의 클러스터는 더 조밀하고(tighter), 서로 다른 작업 간의 거리는 더 멀게(further apart) 형성됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구는 단순한 보상 신호만으로는 학습하기 어려운 Context Encoder에 self-supervised proxy task를 제공함으로써 표현 학습의 품질을 획기적으로 높였다. 특히 Meta-World에서 더 큰 성능 향상이 나타난 점은 주목할 만하다. MuJoCo Controls는 작업 분포가 상대적으로 단순하여 Policy 자체가 병목(bottleneck)인 반면, Meta-World는 작업의 다양성과 복잡성이 높아 **작업 표현 학습(Task Representation Learning)**이 성능의 핵심 요소임을 시사한다.

### 한계 및 논의
-   **계산 비용**: TCL 도입으로 인해 데이터 증강(window cropping) 및 추가적인 gradient 계산이 필요하며, 이로 인해 Meta-World 기준 학습 시간이 약 7.97시간에서 10.16시간으로 증가하였다.
-   **특정 환경의 저조**: Ant-Goal-2D 등 일부 환경에서는 성능이 낮게 나타났는데, 이는 저자들이 PEARL의 기존 성능을 재현하는 데 어려움을 겪었기 때문으로 분석된다.
-   **가정**: 궤적의 윈도우가 작업의 특성을 충분히 담고 있다는 가정하에 작동하며, 윈도우 크기($W$)와 같은 하이퍼파라미터 설정에 따라 성능이 달라질 수 있다.

## 📌 TL;DR

본 논문은 Context-based Meta-RL에서 Context Encoder의 학습 효율을 높이기 위해 **Trajectory Contrastive Learning (TCL)**이라는 자기지도 학습 기법을 제안하였다. 동일 궤적의 윈도우 쌍을 긍정 샘플로, 다른 궤적의 쌍을 부정 샘플로 정의하여 InfoNCE 손실 함수로 학습시키며, 이때 확률 분포 간의 거리를 측정하는 Negative Wasserstein Distance를 도입하였다. 실험 결과, MuJoCo와 Meta-World 벤치마크에서 기존 PEARL 대비 뛰어난 적응 성능과 샘플 효율성을 보였으며, 특히 작업 분포가 복잡한 환경일수록 더 큰 효과를 거두었다. 이 연구는 Meta-RL에서 보상 신호 외의 보조 태스크가 작업 표현 학습에 얼마나 중요한지를 입증하였으며, 향후 복잡한 멀티태스크 적응 연구에 중요한 기초가 될 것으로 보인다.