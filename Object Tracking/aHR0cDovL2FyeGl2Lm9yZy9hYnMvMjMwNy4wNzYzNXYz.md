# CoTracker: It is Better to Track Together
Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht

---

## 🧩 Problem to Solve
이 논문은 동적인 객체와 움직이는 카메라가 포함된 긴 비디오 시퀀스에서 수많은 2D 점들의 **대응 관계를 추정**하는 문제를 다룹니다. 대부분의 기존 점 추적 방식은 점들을 독립적으로 추적하여 통계적 종속성을 간과하는데, 이는 특히 가려지거나 카메라 시야 밖으로 나가는 점들을 추적할 때 정확도와 견고성을 저하시킵니다. 저자들은 추적 대상 점들 간의 종속성을 고려하는 것이 추적 성능을 크게 향상시킬 수 있다고 가정합니다.

## ✨ Key Contributions
*   **공동 점 추적**: 트랜스포머 기반 아키텍처에서 어텐션 메커니즘을 통해 추적되는 점들 간의 정보를 공유하여 수많은 2D 점들을 공동으로 추적하는 개념을 도입했습니다. 이는 추적 정확도와 견고성을 크게 향상시키며, 가려지거나 시야 밖으로 나간 점들을 추적할 수 있도록 합니다.
*   **지원점 (Support Points) 활용**: 사용자에게 요청되지 않은 추가적인 "지원점"들을 추적하여 모델에 더 풍부한 컨텍스트를 제공함으로써 추적 성능을 개선하는 아이디어를 제안했습니다.
*   **프록시 토큰 (Proxy Tokens)을 통한 메모리 복잡도 감소**: 새로운 "프록시 토큰" 개념을 도입하여 모델의 메모리 복잡도를 크게 줄였습니다. 이를 통해 CoTracker는 단일 GPU에서 추론 시 최대 7만 개의 점을 공동으로 동시 추적할 수 있습니다.
*   **언롤드 모델 학습 전략**: 재귀 네트워크처럼 작동하는 윈도우 기반 트래커를 언롤드(unrolled) 방식으로 학습하는 전략을 제안하여, 긴 폐색 상황에서도 장기 추적 성능과 가시성 예측 정확도를 더욱 향상시켰습니다.
*   **최첨단 성능 달성**: 표준 점 추적 벤치마크에서 기존 트래커들을 상당한 마진으로 능가하는 최첨단 성능을 달성했습니다.

## 📎 Related Works
*   **옵티컬 플로우 (Optical Flow)**: 흐름을 순식간의 조밀한 움직임으로 추정하는 분야로, 기존 방법(밝기 불변 방정식, Lucas/Kanade, Horn/Schunck)과 딥러닝 기반 방법(FlowNet, RAFT, Flowformer, GMFlow, Perceiver IO)을 언급합니다. 옵티컬 플로우는 시간에 따른 오차 누적으로 드리프트가 발생할 수 있습니다.
*   **멀티 프레임 옵티컬 플로우 (Multi-frame Optical Flow)**: 옵티컬 플로우를 여러 프레임으로 확장한 연구들(Kalman filtering, VideoFlow, MFT)을 소개하지만, 장기 추적이나 긴 폐색 처리에는 적합하지 않다고 지적합니다.
*   **시각 객체 추적 (Visual Object Tracking)**: 딥러닝 이전의 수동 설계 방식의 공동 추적기, 다중 객체 추적(폐색, 외형 변화, 시간적 사전 정보) 연구, 그리고 시각 객체 추적에 영감을 받은 최근 점 추적기(TAPIR, PIPs) 등을 언급합니다.
*   **임의의 점 추적 (Tracking Any Point, TAP)**: Particle Video (점들을 공동으로 추적), PIPs (독립적 추적, 슬라이딩 윈도우), TAP-Vid (새로운 벤치마크와 기준선), TAPIR (두 단계 피드포워드 추적기), PIPs++ (PIPs의 간소화 버전) 등을 언급합니다. OmniMotion은 테스트 시 최적화 비용이 높습니다.
*   **합성 데이터셋**: 실제 데이터 주석의 어려움으로 인해 TAP-Vid-Kubric, PointOdyssey와 같은 합성 데이터셋이 학습에 활용됨을 설명합니다.

## 🛠️ Methodology
CoTracker는 긴 비디오 시퀀스에서 수많은 2D 점을 추적하기 위한 트랜스포머 기반의 온라인 알고리즘입니다. 짧은 윈도우에서 인과적으로 작동하지만, 재귀 네트워크처럼 언롤드 학습을 통해 장기간 추적을 유지합니다.

1.  **점 추적 정의**:
    *   입력: 비디오 $V = (I_t)_{t=1}^T$ 와 $N$개 점의 초기 위치 및 시작 시간 $(P_i^{t_i}, t_i)_{i=1}^N$.
    *   출력: 모든 유효 시간 $t \geq t_i$ 에 대한 추정된 트랙 위치 $(\hat{P}_i^t = (\hat{x}_i^t, \hat{y}_i^t))$ 와 가시성 플래그 $(\hat{v}_i^t)$.
2.  **트랜스포머 구성**:
    *   **이미지 특징 ($\phi(I_t)$)**: 각 비디오 프레임 $I_t \in \mathbb{R}^{3 \times H \times W}$ 에서 CNN을 사용하여 $d$차원 특징 ($\phi(I_t) \in \mathbb{R}^{d \times \frac{H}{k} \times \frac{W}{k}}$)을 추출합니다. 효율성을 위해 해상도를 $k=4$로 줄이며, $S=4$ 스케일의 다운스케일된 특징 $\phi_s(I_t)$도 활용합니다.
    *   **트랙 특징 ($Q_i^t \in \mathbb{R}^d$)**: 트랙의 외형을 포착하며, 시작 위치에서 샘플링된 이미지 특징으로 초기화되고 네트워크에 의해 업데이트됩니다.
    *   **공간 상관 특징 ($C_i^t \in \mathbb{R}^S$)**: RAFT [58]와 유사하게, 트랙 특징 $Q_i^t$와 현재 추정된 트랙 위치 $\hat{P}_i^t$ 주변의 이미지 특징 $\phi_s(I_t)$을 비교하여 얻습니다. ($S=4, \Delta=3$일 때 차원은 $(2\Delta+1)^2 S = 196$).
    *   **토큰**:
        *   **입력 토큰 $G_i^t$**: 위치, 가시성, 외형, 상관 정보를 인코딩하며, 다음 요소들의 연결로 구성됩니다: $(\hat{P}_i^t - \hat{P}_i^1, \hat{v}_i^t, Q_i^t, C_i^t, \eta(\hat{P}_i^t - \hat{P}_i^1)) + \eta'(\hat{P}_i^1) + \eta'(t)$. (여기서 $\eta, \eta'$는 사인파 위치 인코딩입니다.)
        *   **출력 토큰 $O_i^t$**: 위치 및 외형 업데이트 $(\Delta \hat{P}_i^t, \Delta Q_i^t)$를 포함합니다.
    *   **반복 트랜스포머 적용**: 트랜스포머 $\Psi$를 $M$번 반복하여 트랙 추정치를 점진적으로 개선합니다. 각 반복마다 $\hat{P}^{(m+1)} = \hat{P}^{(m)} + \Delta \hat{P}$ 및 $Q^{(m+1)} = Q^{(m)} + \Delta Q$로 업데이트됩니다. 가시성 $\hat{v}$는 마지막 트랜스포머 적용 후 한 번만 시그모이드 활성화 함수 $\sigma(WQ^{(M)})$를 통해 예측됩니다.
3.  **트랜스포머 아키텍처 및 프록시 토큰**:
    *   트랜스포머 $\Psi$는 시간과 트랙 차원 간의 어텐션 레이어를 교차합니다. 원래 $O(N^2 T^2)$ 복잡도를 가지나, 어텐션 분해 [2]로 $O(N^2 + T^2)$로 줄어듭니다.
    *   **프록시 토큰**: $K$개의 학습된 고정 프록시 토큰을 도입하여, 트랙 간 어텐션을 $K$개의 프록시와 트랙 간의 효율적인 교차 어텐션으로 대체합니다. 이로 인해 복잡도가 $O(NK + K^2 + T^2)$로 감소하며, $K \ll N$일 때 대규모 트랙 추적을 가능하게 합니다. 프록시 토큰은 Registers [17]와 유사하게 작동합니다.
4.  **윈도우 추론 및 언롤드 학습**:
    *   **윈도우 추론**: 임의로 긴 비디오 ($T' > T$)를 처리하기 위해, 비디오를 $J = \lceil 2T'/T - 1 \rceil$개의 길이 $T$ 윈도우로 분할하고 $T/2$ 프레임만큼 겹치게 합니다. 이전 윈도우의 마지막 $T/2$ 프레임 예측을 다음 윈도우의 초기 예측으로 사용합니다.
    *   **언롤드 학습**: 윈도우 트랜스포머가 재귀 네트워크처럼 작동하므로, 언롤드 방식으로 학습합니다. 손실 함수는 반복된 트랜스포머 적용과 윈도우에 걸쳐 합산된 트랙 예측 오차 ($L_1(\hat{P}, P) = \sum_{j=1}^J \sum_{m=1}^M \gamma^{M-m} \| \hat{P}^{(m,j)} - P^{(j)} \|$)와 가시성 플래그의 교차 엔트로피 ($L_2(\hat{v}, v) = \sum_{j=1}^J \text{CE}(\hat{v}^{(M,j)}, v^{(j)})$)를 포함합니다. 이를 통해 모델은 단일 윈도우보다 긴 폐색 및 카메라 시야 밖 점을 추적할 수 있습니다.
5.  **지원점 (Support Points)**: 단일 점 추적 시에도 컨텍스트를 제공하기 위해 추가 지원점을 활용합니다.
    *   **"전역" 전략**: 이미지 전체에 규칙적인 격자 형태의 지원점.
    *   **"지역" 전략**: 추적하려는 점 주변에 격자 형태의 지원점.
    *   **"전역 & 지역" 전략**: 두 전략을 결합하여 사용합니다.

## 📊 Results
*   **TAP-Vid 벤치마크 (DAVIS First/Strided, RGB-Stacking First)**:
    *   CoTracker는 TAPIR, PIPs++ 등 기존 SOTA 모델들을 모든 TAP-Vid 벤치마크에서 상당한 성능 차이로 능가했습니다. 특히 가시점 정확도 ($\delta_{vis}^{avg} \uparrow$) 및 평균 자카드 지수 (AJ $\uparrow$)에서 높은 점수를 기록했습니다.
    *   합성 데이터셋인 Kubric으로 학습했음에도 불구하고, 실제 DAVIS 데이터셋에 대한 뛰어난 일반화 능력을 보였습니다.
*   **PointOdyssey 벤치마크**:
    *   CoTracker는 $\delta_{avg} \uparrow$, $\delta_{vis}^{avg} \uparrow$, $\delta_{occ}^{avg} \uparrow$ 및 Survival rate $\uparrow$ 모든 지표에서 PIPs++를 포함한 기존 모델들을 앞섰습니다. 특히, 단 8프레임의 윈도우 사이즈를 사용했음에도 불구하고 PIPs++ (128프레임 윈도우)보다 더 높은 Survival rate(55.2%)를 달성하여 언롤드 학습의 효과를 입증했습니다.
*   **Dynamic Replica 벤치마크**:
    *   CoTracker는 가려진 점의 정확도 ($\delta_{occ}^{avg} \uparrow$)에서 다른 모델들과 뚜렷한 격차를 보이며 가장 높은 점수를 기록했습니다 (37.6). 이는 공동 추적이 폐색된 점의 움직임을 이해하는 데 매우 효과적임을 보여줍니다.
*   **어블레이션 연구**:
    *   **공동 추적의 중요성 (표 4)**: 공동 추적은 특히 가려진 점의 정확도 ($\delta_{occ}^{avg}$)를 28.8에서 37.6으로 (+30.6%) 크게 개선했습니다.
    *   **언롤드 학습의 중요성 (표 5)**: 언롤드 학습을 사용하지 않을 경우 AJ 점수가 18점 감소하여, 장기 추적에 언롤드 학습이 필수적임을 보여주었습니다.
    *   **프록시 토큰의 효과 (표 6)**: 프록시 토큰을 사용하면 동일한 메모리(80GB GPU)에서 7.4배 더 많은 점을 추적할 수 있으며, 추론 시간도 최대 7배 빨라졌습니다. 64개의 프록시 토큰이 최적의 성능을 보였습니다.
    *   **최적 지원점 구성 (표 7)**: 지원점을 추가하는 모든 구성이 성능 향상에 도움이 되었으며, 전역 및 지역 컨텍스트를 결합한 경우가 가장 좋은 성능을 보였습니다.

## 🧠 Insights & Discussion
*   **공동 추적의 이점**: CoTracker는 점들 간의 통계적 종속성을 활용하는 공동 추적 방식을 통해 기존 독립 추적 방식의 한계를 극복했습니다. 이는 특히 폐색되거나 시야 밖으로 나가는 점들의 추적 정확도를 크게 향상시키며, 장면의 움직임을 더 잘 이해하는 데 기여합니다.
*   **언롤드 학습의 강력함**: 언롤드 학습 전략은 모델이 훈련 시 사용된 시퀀스 길이보다 10배 이상 긴 기간 동안에도 정보를 효과적으로 전파하고, 긴 폐색을 통해 점들을 추적할 수 있게 합니다. 이는 CoTracker가 온라인 추적 알고리즘임에도 불구하고 장기 추적에서 탁월한 성능을 보이는 핵심 요인입니다.
*   **프록시 토큰을 통한 확장성**: 프록시 토큰은 트랜스포머의 계산 복잡도를 효율적으로 낮춰, 단일 GPU에서 거의 준밀도(quasi-dense)에 가까운 매우 많은 수의 점들을 동시 추적할 수 있도록 합니다. 이는 CoTracker의 실제 응용 가능성을 크게 확장시킵니다.
*   **지원점의 컨텍스트 제공 효과**: 사용자가 요청한 점 외에 추가적인 지원점을 추적하는 것이 컨텍스트를 풍부하게 하여 추적 정확도를 높이는 데 도움이 됩니다. 특히 대상 점 주변의 지역적 컨텍스트와 전역적 컨텍스트를 함께 사용하는 것이 모델이 카메라 및 객체 움직임을 모두 추적하는 데 가장 효과적이었습니다.
*   **제한 사항**:
    *   순수하게 합성 데이터로 학습되었기 때문에 반사나 그림자와 같은 복잡한 실제 장면에서는 일반화 능력이 제한될 수 있습니다. CoTracker는 때때로 그림자를 객체와 함께 추적하는 경향이 있는데, 이는 응용 프로그램에 따라 장점이 될 수도, 단점이 될 수도 있습니다.
    *   여러 샷으로 구성된 불연속적인 비디오(예: TAP-Vid-Kinetics)에서는 CoTracker의 연속 비디오 설계 가정이 충족되지 않아, 특정 매칭 모듈을 갖춘 오프라인 방법인 TAPIR와의 성능 격차가 줄어들 수 있습니다.

## 📌 TL;DR
CoTracker는 점들 간의 **통계적 종속성을 활용**하여 **수많은 2D 점을 공동으로 추적**하는 최첨단 트랜스포머 기반 모델입니다. **프록시 토큰**을 도입하여 단일 GPU에서 7만 개 이상의 점을 효율적으로 추적하며, **언롤드 학습**을 통해 긴 폐색 및 시야 밖 점 추적에서 탁월한 성능을 발휘합니다. 또한 **지원점**을 사용하여 추적 컨텍스트를 풍부하게 합니다. 주요 벤치마크에서 SOTA 성능을 달성했으며, 특히 **가려진 점 추적**에서 강력한 강점을 보입니다.