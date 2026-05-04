# TrackingMiM: Efficient Mamba-in-Mamba Serialization for Real-time UAV Object Tracking

Bingxi Liu, Calvin Chen, Junhao Li, Guyang Yu, Haoqian Song, Xuchen Liu, Jinqiang Cui, and Hong Zhang (2025)

## 🧩 Problem to Solve

무인 항공기(UAV) 객체 추적 시스템에서는 고고도에서 캡처된 이미지 시퀀스를 실시간으로 처리해야 하며, 최소 30 FPS 이상의 프레임 속도가 요구된다. 그러나 UAV는 온보드 하드웨어의 전력 및 계산 자원이 매우 제한적이라는 물리적 제약이 있다.

기존의 Vision Transformer(ViT) 기반 추적기는 높은 정확도를 보이지만, Self-attention 메커니즘의 이차 복잡도(quadratic complexity)로 인해 계산 비용과 메모리 요구량이 매우 크다. 반면, Discriminative Correlation Filters(DCF)나 CNN 기반 방식은 효율적일 수 있으나 복잡한 동적 환경에서의 강건성이나 정확도가 떨어진다는 한계가 있다.

특히, 최근 주목받는 State Space Model(SSM)인 Mamba를 추적 작업에 적용하려는 시도가 있었으나, 기존 Mamba 기반 방식들은 Mamba의 스캐닝 메커니즘이 시간적 연속성(temporal continuity)을 충분히 반영하지 못해 시간적 일관성이 부족하다는 문제가 있다. 본 논문의 목표는 이러한 계산 효율성과 시간적 일관성 문제를 동시에 해결하여, 저사양 하드웨어(예: 4GB GPU 메모리)에서도 실시간 동작이 가능한 고성능 UAV 추적 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문은 Mamba의 선형 복잡도와 효율성을 유지하면서 UAV 추적의 특수성을 반영한 **TrackingMiM** 프레임워크를 제안한다. 핵심 아이디어는 다음과 같다.

1.  **Mamba-in-Mamba (MiM) 구조**: 공간적 특징 추출과 시간적 연속성 모델링을 분리하여 처리하는 중첩(nested) 구조를 설계하였다. 내부 Mamba는 프레임 내(intra-frame)의 세부 공간 특징을 학습하고, 외부 Mamba는 프레임 간(inter-frame)의 시간적 일관성을 강제한다.
2.  **시간 직렬화 스캐닝 (Time Serialization Scanning)**: 동적 환경에서 시스템의 시간적 진화를 명시적으로 캡처하기 위해 Mamba의 스캔 경로를 체계적으로 재배열하여 시간적 인지 능력을 향상시켰다.
3.  **쿼리 기반 검색 증강 추적 (Query-Based Retrieval Augmented Tracking, RAT)**: 과거의 추적 특징을 저장하는 메모리 코퍼스를 구축하고, 현재 쿼리와 가장 유사한 Top-K 특징을 검색하여 추적 과정에 통합함으로써 폐쇄(occlusion)나 급격한 외형 변화 상황에서의 강건성을 높였다.

## 📎 Related Works

논문은 UAV 추적 알고리즘을 세 가지 범주로 분류하여 설명한다.

-   **DCF-based**: FFT를 통한 계산 효율성이 매우 높고 CPU 구현에 적합하지만, 수작업 특징(hand-crafted features)에 의존하여 복잡한 환경에서의 적응력이 낮다.
-   **CNN-based**: 적응적인 특징 학습으로 정확도가 높으나, 합성곱 연산의 계산 비용이 커 실시간 적용에 어려움이 있다. 이를 해결하기 위해 네트워크 경량화 및 가지치기(pruning) 연구가 진행되었으나 정확도 손실이 발생한다.
-   **ViT-based**: Self-attention을 통해 장거리 의존성을 효과적으로 모델링하여 SOTA 성능을 달성했지만, 높은 모델 복잡도로 인해 자원 제한적인 UAV 플랫폼 배포가 어렵다.

본 연구는 Mamba의 선형 계산 복잡도를 활용하여 ViT의 성능과 DCF/CNN의 효율성 사이의 트레이드-오프를 해결하고자 하며, 특히 기존 Mamba 기반 모델들이 간과한 '시간적 연속성' 문제를 해결한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Preliminary: SSM and Mamba
Mamba는 선형 시불변 시스템을 모델링하는 State Space Model(SSM)에 기반한다. 입력 시퀀스 $x(t)$는 다음과 같은 연속 시간 상태 방정식으로 정의된다.

$$\dot{h}(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

이를 이산화(discretization)하여 수치 계산이 가능하게 만들며, Mamba는 선택적 상태 공간(selective state space)을 통해 입력에 따라 매개변수를 동적으로 조정함으로써 장거리 의존성을 선형 복잡도로 캡처한다.

### 2. Mamba-in-Mamba Architecture
전체 파이프라인은 입력 이미지를 패치 토큰으로 변환한 후, 중첩된 Mamba 블록을 통과시키는 구조이다.

**토큰화 (Tokenization):**
템플릿 프레임 $X_0$와 검색 프레임 시퀀스 $X_t$를 $K \times K$ 크기의 패치로 나누어 토큰화한다. 여기에 학습 가능한 공간 위치 임베딩 $e_s$와 시간 위치 임베딩 $e_t$를 더해 입력 벡터 $P$를 생성한다.
$$P = [P_{0,p}, P_{t,p}] + e_s + e_t$$

**MiM 블록의 작동 순서:**
1.  **내부 공간 Mamba (Inner Spatial Mamba)**: 각 프레임 내에서 공간적 상호작용을 먼저 학습한다.
    -   **Template-first Spatial Scanning**: 템플릿 특징을 먼저 계산하고, 이를 기반으로 검색 프레임의 스캔을 수행하여 템플릿 정보를 효율적으로 통합한다.
2.  **검색 증강 추적 (RAT)**: 공간 Mamba와 시간 Mamba 사이에 위치하여 타겟 인지 능력을 강화한다.
3.  **외부 시간 Mamba (Outer Temporal Mamba)**: 추출된 특징들을 시간 축으로 전파하며, 양방향 스캐닝 전략을 통해 프레임 간의 장거리 의존성을 인코딩한다.

### 3. Time Serialization Scanning
단순한 공간 스캔은 정적인 상태를 가정하지만, UAV 추적과 같은 동적 시스템에서는 $\frac{dP}{dt} \neq 0$인 상황이 발생한다. 이를 해결하기 위해 시간적 분해능을 고려한 이산 샘플링 방식을 도입하여 시간적 진화를 명시적으로 재구성한다.
$$P_t^i = P_t^i + \sum_{i=0}^t \Delta t \cdot K_t^i \cdot \frac{dP_t}{dt}$$

### 4. Retrieval Augmented Tracking (RAT)
과거의 특징을 활용해 현재의 타겟 위치를 가이드하는 메커니즘이다.
-   **코퍼스 업데이트**: 새로운 특징 $e_q$가 기존 메모리 $C$의 항목들과 코사인 유사도가 임계값 $\tau = 0.8$보다 작을 때만 추가하여 중복을 방지한다.
-   **검색 및 융합**: 현재 쿼리 $e_q$와 가장 유사한 Top-K 특징을 검색하고, 이들의 평균(mean fusion)을 구하여 증강된 특징 $e_a$를 생성한다.
-   **추적 어텐션 (Track Attention)**: $e_a$를 쿼리(Query)로 사용하고, 공간 Mamba의 출력을 키-값(Key-Value)으로 사용하는 교차 어텐션(Cross-attention)을 수행하여 최종 잠재 표현 $\hat{h}$를 업데이트한다.
$$\hat{h} = \text{softmax}\left(\frac{Q_a K^\top}{\sqrt{d_k}}\right) V$$

## 📊 Results

### 1. 실험 설정
-   **데이터셋**: UAV123, UAV123@10fps, VisDrone2018, UAVDT, DTB70 등 5개 벤치마크 사용.
-   **비교 대상**: DCF, CNN, ViT 기반의 대표적인 경량 추적기 25종.
-   **평가 지표**: 정밀도(Precision), 성공률(Success Rate), 처리 속도(FPS).

### 2. 정량적 결과
-   **정확도**: TrackingMiM은 평균 정밀도 $86.3\%$와 성공률 $66.1\%$를 기록하여 비교 대상 중 가장 높은 성능을 보였다. 특히 기존 SOTA였던 Aba-ViTrack($85.4\%$ Prec, $64.9\%$ Succ)보다 향상된 수치를 보였다.
-   **속도**: GPU 기준 **268.3 FPS**라는 매우 빠른 속도를 달성하였으며, CPU에서도 **97.2 FPS**를 기록하여 실시간 요구사항(30 FPS)을 충분히 만족한다. 이는 대부분의 ViT 기반 모델보다 월등히 빠르며, 일부 고속 CNN 모델(DRCI 등)과 대등한 수준이다.

### 3. 분석 및 소거 연구 (Ablation Study)
-   **구성 요소 영향**: 시간적 스캐닝을 제거할 경우 정밀도가 $2.0\%$ 감소하며, RAT를 제거할 경우 정밀도가 $2.9\%$ 감소한다. 두 기능을 모두 포함했을 때 가장 높은 성능 향상($+4.0$ Prec)이 나타났다.
-   **Plug-and-Play 검증**: 제안한 RAT 모듈을 다른 6종의 기존 추적기에 통합한 결과, 모든 모델에서 정밀도 $+1.0$ 이상, 성공률 $+0.9$ 이상의 성능 향상이 관찰되었다.
-   **하이퍼파라미터**: 패치 크기(36), 레이어 깊이(24), 시간 윈도우 크기(8), 검색 수($K=7$)가 속도와 정확도의 최적 균형점임을 확인하였다.

## 🧠 Insights & Discussion

**강점:**
본 논문은 Mamba의 선형 복잡도를 영리하게 활용하여, ViT 수준의 높은 정확도를 유지하면서도 DCF/CNN 수준의 실시간성을 확보하였다. 특히, 단순히 Mamba를 적용한 것이 아니라 'Mamba-in-Mamba'라는 계층적 구조와 'RAT'라는 외부 메모리 메커니즘을 결합하여, Mamba의 고유한 약점인 시간적 연속성 결여 문제를 효과적으로 해결하였다.

**한계 및 논의사항:**
-   **하이퍼파라미터 민감도**: 실험 결과에서 보듯 패치 크기나 레이어 깊이에 따라 FPS 변동이 매우 심하다. 이는 실제 하드웨어 배포 시 타겟 기기의 성능에 맞춰 세밀한 튜닝이 필수적임을 시사한다.
-   **메모리 관리**: RAT 모듈이 코사인 유사도 기반으로 코퍼스를 업데이트하지만, 장시간 추적 시 메모리 코퍼스의 크기가 계속 증가할 가능성이 있으며 이에 대한 명시적인 최대 크기 제한이나 교체 전략에 대한 언급은 부족하다.
-   **비판적 해석**: CPU 성능이 97.2 FPS로 매우 높게 측정되었는데, 이는 Mamba의 효율성 덕분이기도 하지만 입력 해상도와 패치 크기 설정이 계산량을 크게 줄였기 때문일 수 있다. 더 높은 해상도에서의 성능 유지 여부에 대한 추가 검증이 필요해 보인다.

## 📌 TL;DR

본 논문은 UAV 실시간 객체 추적을 위해 **중첩된 Mamba 구조(Mamba-in-Mamba)**와 **검색 증강 추적(RAT)** 메커니즘을 결합한 **TrackingMiM**을 제안한다. 이를 통해 ViT의 고질적인 계산 복잡도 문제를 해결하고, 시간적 연속성을 확보함으로써 **SOTA 수준의 정확도(Prec 86.3%)**와 **압도적인 추론 속도(268.3 FPS)**를 동시에 달성하였다. 특히 제안된 RAT 모듈은 다른 추적기에도 적용 가능한 plug-and-play 형태의 모듈로서, 향후 자원 제한적인 환경의 실시간 비전 추적 연구에 중요한 기준점이 될 것으로 보인다.