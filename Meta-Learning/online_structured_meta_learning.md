# Online Structured Meta-learning

Huaxiu Yao, Yingbo Zhou, Mehrdad Mahdavi, Zhenhui Li, Richard Socher, Caiming Xiong (2020)

## 🧩 Problem to Solve

본 논문은 온라인 환경에서 새로운 태스크에 빠르게 적응하기 위한 Online Meta-learning의 한계를 해결하고자 한다. 기존의 온라인 메타 러닝 알고리즘들은 모든 태스크가 공유하는 하나의 전역적 메타 러너(globally-shared meta-learner)를 학습하는 방식에 의존한다. 그러나 실제 환경에서 마주하는 태스크들은 서로 성격이 매우 다른 이질적인(heterogeneous) 정보들을 포함하고 있을 가능성이 높다.

만약 태스크 분포가 서로 겹치지 않는 이질적인 모드(disjoint modes)를 가지고 있다면, 단일한 전역 메타 러너로는 모든 모드의 정보를 충분히 포괄할 수 없으며 이는 결국 최적이지 않은(sub-optimal) 성능으로 이어진다. 따라서 본 연구의 목표는 이질적인 태스크 분포 하에서도 효율적으로 지식을 공유하고 적응할 수 있는 구조화된 메타 러닝 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간의 지식 조직화 방식과 계층적 특징 표현(hierarchical feature representation)에서 영감을 얻어, 메타 러너를 여러 개의 지식 블록(knowledge blocks)으로 구성된 '메타 계층 그래프(meta-hierarchical graph)' 형태로 분리(disentangle)하는 것이다.

주요 기여 사항은 다음과 같다.
첫째, 이질적인 분포 설정에서의 온라인 메타 러닝 문제를 공식화하고, 메타 계층 그래프를 유지하는 OSML(Online Structured Meta-learning) 프레임워크를 제안하였다.
둘째, 새로운 태스크가 주어졌을 때 가장 관련성이 높은 지식 블록들을 선택하여 '메타 지식 경로(meta-knowledge pathway)'를 자동으로 구축하고, 필요한 경우 새로운 블록을 생성하여 미지의 지식을 확장하는 메커니즘을 도입하였다.
셋째, 구축된 메타 계층 구조가 태스크 간의 구조적 정보를 캡처함으로써 모델의 해석 가능성(interpretability)을 높였음을 입증하였다.

## 📎 Related Works

기존의 메타 러닝은 주로 모든 태스크가 동일한 정지 분포(stationary distribution)에서 샘플링되었다고 가정한다. MAML과 같은 최적화 기반 메타 러닝은 전역적인 초기 파라미터를 학습하여 빠른 적응을 돕는다. Finn 등이 제안한 FTML(Follow The Meta Leader)은 이를 온라인 설정으로 확장하여 메타 러너를 지속적으로 업데이트하지만, 여전히 전역 공유 메타 러너를 사용한다는 한계가 있다.

이질적인 태스크 분포를 해결하기 위해 일부 연구는 태스크별 정보로 메타 러너를 변조(modulating)하거나, Dirichlet Process Mixture(DPM)와 같은 계층적 베이지안 모델을 적용하였다. 그러나 변조 방식은 잘 학습된 태스크 표현 네트워크가 필요하여 온라인 설정에서는 비실용적이며, DPM 방식은 이질적인 태스크에 대해 완전히 새로운 메타 러너를 생성해야 하므로 지식 공유의 유연성이 떨어진다. OSML은 블록 단위의 설계를 통해 지식의 탐색(exploration)과 활용(exploitation) 사이의 균형을 맞춤으로써 이러한 한계를 극복한다.

## 🛠️ Methodology

OSML은 메타 러너를 여러 층(layer)으로 구성하고, 각 층에 여러 개의 지식 블록을 배치한 그래프 구조로 정의한다. 전체 프로세스는 크게 '메타 지식 경로 구축'과 '지식 블록 업데이트'의 두 단계로 나뉜다.

### 1. 메타 지식 경로 구축 (Meta-knowledge Pathway Construction)

새로운 태스크 $T_t$가 도착하면, 모델은 각 층에서 가장 관련성이 높은 지식 블록을 찾아 경로를 생성한다. 이때 효율적인 탐색을 위해 미분 가능한(differentiable) 방식으로 이 과정을 완화하여 처리한다.

층 $l$에서의 입력 표현 $g_{l-1,t}$에 대해, 출력 $g_{l,t}$는 다음과 같이 계산된다.

$$g_{l,t} = \frac{\sum_{b_l=1}^{B_l+1} \exp(o^{b_l}) M_t(w^{0b_l,t})(g_{l-1,t})}{\sum_{b'_l=1}^{B_l+1} \exp(o^{b'_l})}$$

여기서 $o$는 각 지식 블록의 중요도를 나타내는 계수이며, $M_t$는 서포트 셋 $D_{supp}^t$를 이용한 내부 업데이트(inner update) 과정을 의미한다.

$$M_t(w^{0b_l,t}) = w^{0b_l,t} - \alpha \nabla_{w^{0b_l,t}} L(w^{0,t}, D_{supp}^t)$$

이후 쿼리 셋 $D_{query}^t$를 사용하여 초기 파라미터 $w_{0,t}$와 중요도 계수 $o$를 메타 업데이트한다.

$$w_{0,t} \leftarrow w_{0,t} - \beta_1 \nabla_{w_{0,t}} L(w_t, o; D_{query}^t)$$
$$o \leftarrow o - \beta_2 \nabla_o L(w_t, o; D_{query}^t)$$

최종적으로 각 층에서 중요도 $o^{b_l}$가 가장 큰 블록 $b^*_l = \arg \max o^{b_l}$을 선택하여 메타 지식 경로 $\{w^{0b^*_1,t}, \dots, w^{0b^*_L,t}\}$를 형성한다.

### 2. 지식 블록 메타 업데이트 (Knowledge Block Meta-Updating)

선택된 경로상의 블록들은 새로운 태스크의 정보를 반영하여 업데이트된다. 계산 효율성을 위해 2차 미분 대신 1차 근사(first-order approximation)를 사용하며, 태스크 버퍼 $B$에서 해당 블록을 공유하는 이전 태스크 $T_k$들을 샘플링하여 업데이트를 수행한다.

$$\text{update: } w^{0b^*_l,t} \leftarrow w^{0b^*_l,t} - \beta_3 \sum_{k=1}^K \nabla_{w^{b_l,k}} L(w^k; D_{query}^k)$$

마지막으로, 업데이트된 메타 지식 경로를 서포트 셋과 쿼리 셋을 합친 $D_{all}^t$로 파인튜닝하여 해당 태스크에 최적화된 파라미터를 얻는다.

$$w^{b^*_1,t} = w^{0b^*_l,t} - \beta_5 \nabla_w L(w; D_{supp}^t \oplus D_{query}^t)$$

## 📊 Results

### 실험 설정

- **데이터셋**:
  - Homogeneous: Rainbow MNIST (색상, 크기, 각도가 변형된 56개 태스크).
  - Heterogeneous: Multi-filtered mini-Imagenet (Blur, Night, Pencil 필터 적용), Meta-dataset (Flower, Fungi, Aircraft 포함).
- **비교 대상(Baselines)**: Non-transfer (NT), Fine-tune (FT), FTML, DPM, HSML.
- **평가 지표**: 정확도(Accuracy) 및 평균 순위(Average Ranking, AR).

### 주요 결과

- **정량적 성과**: OSML은 모든 데이터셋에서 다른 베이스라인보다 높은 정확도를 보였으며, 특히 이질적인 태스크 분포(Multi-filtered mini-Imagenet, Meta-dataset)에서 FTML 대비 월등한 성능 향상을 보였다.
- **모델 용량 분석**: 단순히 모델의 파라미터 수를 늘린 버전(NT-Large, FT-Large, FTML-Large)과 비교했을 때, OSML의 성능 향상은 단순히 용량이 커졌기 때문이 아니라 구조화된 지식 활용 능력에서 기인함이 확인되었다.
- **학습 효율성**: Rainbow MNIST 실험에서 OSML은 목표 정확도(90%)에 도달하기 위해 필요한 샘플 수가 다른 방법들보다 적어, 더 빠른 학습 효율성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 전역적 공유 모델의 한계를 '지식 블록의 분리'와 '경로 선택'이라는 구조적 접근으로 해결하였다. 특히, Meta-dataset 실험에서 특정 지식 블록이 특정 도메인(예: Aircraft)에 의해 지배되거나, 유사한 도메인(예: Flower와 Fungi)끼리 블록을 공유하는 경향이 발견되었다. 이는 OSML이 태스크 간의 유사성과 차이점을 명시적으로 캡처하고 있음을 보여주며, 모델의 내부 작동 방식을 이해할 수 있는 해석 가능성을 제공한다.

다만, 본 논문에서는 태스크 버퍼의 효율적인 관리 방식이나 제안된 구조의 이론적인 일반화 성능 분석에 대해서는 명시적으로 다루지 않았으며, 이를 향후 연구 과제로 남겨두었다. 또한 실제 실시간 시스템에 적용하기 위해서는 지식 블록의 적절한 개수($B_l$)를 결정하는 하이퍼파라미터 튜닝이 중요할 것으로 판단된다.

## 📌 TL;DR

OSML은 온라인 메타 러닝에서 모든 태스크가 하나의 모델을 공유함으로써 발생하는 성능 저하를 막기 위해, 메타 러너를 계층적 지식 블록 구조로 설계한 프레임워크이다. 태스크별로 최적의 지식 경로를 선택하고 업데이트함으로써 이질적인 태스크 분포에서도 유연한 지식 공유와 빠른 적응이 가능함을 입증하였으며, 이는 향후 자율 주행이나 질병 예측과 같이 복잡하고 다양한 환경에 적응해야 하는 실세계 응용 분야에 중요한 기여를 할 수 있다.
