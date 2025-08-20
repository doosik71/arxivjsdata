# DYNAMIC TEMPLATE SELECTION THROUGH CHANGE DETECTION FOR ADAPTIVE SIAMESE TRACKING
Madhu Kiran, Le Thanh Nguyen-Meidine, Rajat Sahay, Rafael Menelau Oliveira E Cruz, Louis-Antoine Blais-Morin, Eric Granger

### 🧩 해결하고자 하는 문제

*   실시간 단일 객체 추적(SOT)은 객체 외형 변화와 변형으로 인해 도전적인 과제입니다.
*   적응형 추적(Adaptive tracking) 방법은 온라인 학습을 위해 추적기가 수집한 표본을 사용하지만, 모든 수집된 표본으로 학습하면 '파국적 망각(catastrophic forgetting)'으로 인해 모델이 손상될 수 있습니다.
*   기존 적응형 Siameses 추적기는 모델을 자주 업데이트하여 (심지어 불필요한 경우에도) 노이즈가 통합되어 추적 모델의 효율성이 저해됩니다.
*   개념 드리프트(concept drift)와 파국적 망각을 효율적으로 처리할 수 있는 강력한 온라인 적응 전략이 필요합니다.

### ✨ 주요 기여

*   SOT를 온라인 증분 학습 문제로 공식화합니다.
*   템플릿 손상을 방지하기 위한 동적 샘플 선택 및 메모리 리플레이(memory replay)를 위한 새로운 방법을 제안합니다.
*   객체 외형의 점진적 변화를 감지하고, 필요한 경우에만 온라인 적응을 유발하는 변화 감지(change detection) 메커니즘을 도입합니다.
*   메모리 리플레이를 위한 다양성 있는 보조 버퍼(auxiliary buffer)를 유지하기 위해 엔트로피 기반 샘플 선택 전략을 제안하여 파국적 망각 문제를 완화합니다.
*   제안된 방법은 모델 적응을 위해 온라인 학습을 활용하는 모든 객체 추적 알고리즘에 통합될 수 있습니다.
*   기존 최신 적응형 Siameses 추적기에 통합 시 평균 2%의 AUC 정확도 향상과 추적 속도 개선을 달성하며, 비용 효율성과 핵심 구성 요소의 기여도를 입증합니다.

### 📎 관련 연구

*   **Siameses 추적:** SINT, SiamFC, SiamRPN, ATOM, DiMP 등과 같은 Siameses 네트워크 기반 추적기들이 있습니다.
*   **추적을 위한 온라인 학습:** MDNet, TCNN과 같이 계산 비용이 높은 방법과 DiMP, PrDiMP, TrDiMP와 같이 효율적인 최적화 방법을 사용한 최신 추적기들이 포함됩니다. 이동 평균(moving average), 생성 모델(generative models), LSTM 기반 템플릿 추정, SGD(Stochastic Gradient Descent) 등 다양한 온라인 업데이트 전략이 연구되었습니다.
*   **온라인 증분 학습:** 순차적 데이터에 대해 새로운 클래스나 분포를 온라인으로 학습하는 문제로, 파국적 망각을 해결하기 위한 정규화(regularization), 구조적(structural), 리허설(rehearsal) 방법 등이 있습니다. 특히 리허설 방법 중 메모리 리플레이를 사용하는 Gradient Episodic Memory (GEM) 및 Generative replay, 그리고 엔트로피 기반 샘플링([20]에서 영감)이 언급됩니다.
*   **스트리밍 데이터의 변화 감지:** 점진적 변화(gradual change), 급격한 변화(abrupt change), 증분적 변화(incremental change), 반복적 변화(recurrent change) 등 개념 드리프트의 유형과 이를 감지하는 순차적 분석(sequential analysis), 적응형 윈도우(adaptive windowing), 고정 누적 윈도우(fixed cumulative windowing), 통계적 방법(statistical methods) 등이 있습니다. 본 논문에서는 Page-Hinckeley 검정(test)을 사용합니다.

### 🛠️ 방법론

*   **문제 재정의:** 단일 객체 추적(SOT)을 의사 레이블(pseudo labels, 추적기가 생성한 레이블)을 가진 온라인 증분 학습 문제로 간주합니다.
*   **전반적인 프레임워크 (그림 2 참조):**
    1.  **특징 추출:** 추적기가 객체를 찾아내고 특징 추출기 $\Theta$를 사용하여 특징 $\phi_t$를 생성합니다.
    2.  **주 메모리(Main Memory):** 추출된 특징은 FIFO(First-In, First-Out) 방식으로 작동하는 버퍼에 저장됩니다 (예: 50개 인스턴스).
    3.  **온라인 분류기 학습:** 최적화기(optimizer)는 주 메모리의 모든 인스턴스를 사용하여 모델 $f$를 학습합니다. 이 모델은 테스트 이미지 특징 $\phi_{S_t}$와 컨볼루션되어 분류 점수 맵 $S$를 생성하고, 이는 객체 전경과 배경을 구분하여 위치를 특정하는 데 사용됩니다.
    4.  **변화 감지(Change Detection) (제안):** 기존의 매 프레임마다 모델을 업데이트하는 대신, 분류 점수 맵의 최댓값에 Page-Hinckeley 검정(sequential change detector)을 적용하여 개념 드리프트 발생 시에만 분류기 학습을 유발합니다. 이는 불필요한 노이즈 통합을 방지하고 계산 복잡성을 줄입니다.
    5.  **보조 메모리(Auxiliary Memory) (제안):** 오래된 샘플을 저장하기 위한 추가 버퍼(예: 50개)를 유지하여 파국적 망각 문제를 완화합니다.
        *   초기에는 주 메모리에서 나온 샘플로 FIFO 방식으로 채워집니다.
        *   버퍼가 가득 차면 엔트로피 최대화 알고리즘을 사용하여 다양성 있는 샘플을 유지합니다.
        *   **엔트로피 최대화 샘플링:**
            *   각 샘플의 분류기 점수 $S$는 선형 매핑 함수 $D$를 통해 0에서 $Y$까지의 이산적인 레이블로 변환됩니다. 특정 임계값 $\tau$ 미만의 점수를 가진 샘플은 제외됩니다.
            *   목표는 $\phi$와 $Y$의 결합 엔트로피 $H(\phi, Y) = H(\phi|Y) + H(Y)$를 최대화하는 것입니다.
            *   $H(Y)$를 최대화하기 위해 각 이산 레이블에 대해 균등한 수의 샘플을 유지합니다.
            *   $H(\phi|Y)$를 최대화하기 위해, 다수 레이블(majority label)에 속하는 샘플 중 해당 레이블 내에서 다른 샘플들과의 최소 거리가 짧은 샘플을 확률 $(1-d_i) / \sum_j (1-d_j)$에 따라 교체합니다 (여기서 $d_i$는 해당 클래스 내 $x_i$에서 모든 다른 $x_j$까지의 최소 거리).
    6.  **리플레이를 통한 학습:** 추적 중에는 주 메모리의 모든 샘플과 보조 메모리에서 무작위로 추출된 $n$개의 샘플이 분류기 학습을 위한 배치로 사용됩니다. 주 메모리는 현재 표현을 학습하고, 보조 메모리 샘플은 파국적 망각을 방지합니다.

### 📊 결과

*   **데이터셋:** OTB-100, UAV123, LaSOT, TrackingNet.
*   **평가 지표:** 성공도(Success plot)의 AUC(Area Under Curve).
*   **통합 대상 추적기:** DiMP, PrDiMP, TrDiMP.
*   **Ablation 연구 (LaSOT 데이터셋):**
    *   **변화 감지의 효과:** "DiMP with CD" (AUC 57.8)가 "DiMP Random" (55.4), "DiMP Periodic" (56.8/56.5/55.6), 기본 "DiMP" (57.1)보다 우수한 성능을 보였습니다. 이는 "필요할 때만 학습"하는 전략의 효율성을 입증합니다.
    *   **보조 메모리의 효과:** "DiMP with CD + Class. Score Discretised" (AUC 58.6)가 "DiMP with CD + Random Replacement" (57.9) 및 "DiMP with CD + Density Replacement" (58.2)보다 높은 AUC를 달성했습니다. 이는 제안된 엔트로피 기반 샘플링이 다양성을 효과적으로 유지함을 보여줍니다.
*   **전반적인 성능:**
    *   제안된 ADiMP (DiMP 기반)와 APrDiMP (PrDiMP 기반), ATrDiMP (TrDiMP 기반)는 모든 데이터셋에서 해당 베이스라인 추적기보다 일관되게 AUC 정확도를 향상시켰습니다.
    *   예를 들어, ADiMP는 LaSOT에서 DiMP50 대비 약 1.6% (56.9 $\rightarrow$ 58.6), TrackingNet에서 약 1.5% (74 $\rightarrow$ 75.5) 향상.
    *   APrDiMP는 LaSOT에서 PrDiMP50 대비 약 1.7% (59.8 $\rightarrow$ 61.5), TrackingNet에서 약 1.3% (75.8 $\rightarrow$ 77.1) 향상.
    *   ATrDiMP는 LaSOT에서 TrDiMP 대비 약 6% (63.0 $\rightarrow$ 69.0) 향상.
    *   특히 긴 비디오(UAV123, LaSOT)에서 보조 메모리의 이점이 더 크게 나타났습니다.
*   **추적 프레임 속도 (FPS):**
    *   ADiMP (44 FPS)는 DiMP (38 FPS)보다 빠릅니다.
    *   APrDiMP (33 FPS)는 PrDiMP (28 FPS)보다 빠릅니다.
    *   ATrDiMP (29 FPS)는 TrDiMP (25 FPS)보다 빠릅니다.
    *   이는 분류기 학습을 덜 자주(요구에 따라) 수행함으로써 달성된 성능 개선입니다.

### 🧠 통찰 및 논의

*   **의미:** 제안된 접근 방식은 개념 드리프트, 파국적 망각, 노이즈가 있는 의사 레이블 등 SOT의 온라인 학습 문제를 추가적인 계산 오버헤드 없이 효과적으로 해결합니다.
*   **중요성:** 모델 업데이트를 선택적으로 수행하고 다양한 메모리를 유지함으로써 적응형 Siameses 추적기의 강건성과 효율성을 향상시킵니다.
*   **작동 원리:** 변화 감지 메커니즘은 불필요한 업데이트를 방지하여 노이즈 통합과 계산 비용을 줄입니다. 엔트로피 기반 보조 메모리는 저장된 샘플의 다양성을 보장하여 파국적 망각을 효과적으로 완화하고 과거 지식을 보존합니다.
*   이 방법은 온라인 학습을 활용하는 "모든 객체 추적 알고리즘"에 적용 가능하다는 점에서 광범위한 활용 가능성을 가집니다.

### 📌 TL;DR

*   **문제:** 적응형 Siameses 추적기는 잦고 불필요한 업데이트와 노이즈가 있는 샘플로 인해 파국적 망각과 비효율적인 온라인 적응 문제를 겪습니다.
*   **방법:** 객체 외형에 개념 드리프트가 감지될 때만 모델 적응을 유발하는 **변화 감지 메커니즘**을 도입하고, **엔트로피 기반 샘플 선택 전략**을 통해 다양성 있는 보조 메모리 버퍼를 유지하여 파국적 망각을 방지하는 새로운 온라인 증분 학습 프레임워크를 제안합니다.
*   **핵심 발견:** 이 방법을 최신 적응형 Siameses 추적기(DiMP, PrDiMP, TrDiMP)에 통합한 결과, 추적 정확도(AUC 평균 2% 향상)가 크게 개선되었고, 불필요한 업데이트 감소로 추적 속도까지 향상되어 비용 효율성과 강건함이 입증되었습니다.