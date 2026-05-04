# JUICER: Data-Efficient Imitation Learning for Robotic Assembly

Lars Ankile, Anthony Simeonov, Idan Shenfeld, Pulkit Agrawal (2024)

## 🧩 Problem to Solve

본 논문은 로봇 조립(Robotic Assembly) 작업에서 적은 양의 인간 시연 데이터만으로 고성능의 Visuomotor Policy를 학습시키는 문제를 해결하고자 한다. 특히 가구 조립과 같이 긴 호라이즌(Long-horizon)을 가지며, 정밀한 파지(Grasping), 재배향(Reorienting), 그리고 삽입(Insertion) 작업이 요구되는 환경에서 기존의 모방 학습(Imitation Learning, IL)은 막대한 양의 데이터 수집 부담이 있으며, 작은 오차가 치명적인 실패로 이어지는 Covariate Shift 문제에 취약하다는 한계가 있다. 

따라서 본 연구의 목표는 제한된 시연 예산(약 50개의 시연) 하에서 최대한의 성능을 이끌어낼 수 있는 데이터 효율적인 모방 학습 파이프라인인 JUICER를 제안하고, 이를 통해 정밀한 조립 작업을 수행하는 것이다.

## ✨ Key Contributions

JUICER의 핵심 아이디어는 표현력이 뛰어난 정책 아키텍처와 체계적인 데이터 확장 기법을 결합하여, 데이터 부족 문제를 해결하고 병목 지점(Bottleneck regions)에서의 강건성을 확보하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **Diffusion Policy의 도입**: 복잡한 행동 분포를 모델링하기 위해 Conditional Diffusion Model을 채택하고, 미래의 행동들을 묶음(Chunk) 형태로 예측하여 행동의 일관성을 높였다.
2.  **병목 상태 기반 궤적 증강(Trajectory Augmentation)**: 정밀도가 요구되는 병목 상태를 정의하고, 해당 상태에서 인위적으로 이탈했다가 다시 복귀하는 '교정 행동(Corrective actions)' 데이터를 합성하여 데이터셋의 지지 영역(Support)을 확장하였다.
3.  **Collect-and-Infer 루프**: 모델 평가 과정에서 발생한 성공적인 롤아웃(Rollout) 데이터를 다시 학습셋에 포함시키는 반복적 데이터 확장 방식을 통해 추가적인 인간의 노력 없이 데이터 다양성을 확보하였다.
4.  **데이터 효율성 증명**: 제안된 파이프라인을 통해 단 10개의 인간 시연만으로도 상당한 수준의 조립 성능을 달성할 수 있음을 보여주었다.

## 📎 Related Works

본 논문은 기존의 모방 학습 강건성 향상 기법들과 비교하여 다음과 같은 차별점을 가진다.

-   **행동 복제(Behavior Cloning, BC)의 강건성**: 기존 연구들($[9], [10]$ 등)은 상태 공간에 노이즈를 주입하거나 국소적 동역학 모델을 통해 교정 행동을 생성하였다. 그러나 이러한 방식은 주로 객체의 포즈가 명확히 알려진 상태 공간에서 작동하며, RGB 이미지 기반의 visuomotor policy에 직접 적용하기에는 한계가 있다.
-   **Diffusion for Decision-Making**: 최근 Diffusion 모델이 고차원 행동 분포를 학습하는 데 효과적임이 밝혀졌으며, 본 논문은 이를 조립 작업의 긴 호라이즌 문제에 적용하여 MLP 기반 모델보다 뛰어난 성능을 확인하였다.
-   **조립 및 삽입 학습**: InsertionNet이나 Form2Fit과 같은 연구들은 분해(Disassembly) 시퀀스를 역이용하여 삽입 동작을 학습하였다. JUICER는 이 원리를 일반화하여 병목 지점 주변의 데이터 증강에 활용함으로써, 연속적인 저수준 제어(Low-level control) 환경에서도 작동하도록 설계되었다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
JUICER 파이프라인은 다음과 같은 단계로 구성된다:
$\text{소량의 시연 수집} \rightarrow \text{병목 상태 주석(Annotation)} \rightarrow \text{궤적 증강(TA)} \rightarrow \text{Diffusion Policy 학습} \rightarrow \text{성공 롤아웃 수집(CI)} \rightarrow \text{최종 모델 학습}$.

### 2. 궤적 증강 (Trajectory Augmentation)
정밀한 삽입 작업 중 발생하는 작은 오차가 실패로 이어지는 것을 방지하기 위해, 다음과 같은 알고리즘으로 합성 데이터를 생성한다.
-   시연 데이터에서 정밀도가 필요한 **병목 상태($s_{\text{bottleneck}}$)**를 샘플링한다.
-   해당 상태에서 임의의 목표 상태($s_{\text{target}}$)로 이동하는 '분해' 동작을 생성하고 이를 기록한다.
-   기록된 분해 동작을 역순으로 실행하여 $s_{\text{target}}$에서 $s_{\text{bottleneck}}$으로 돌아오는 **교정 행동**을 합성한다.
-   최종 상태가 원래의 병목 상태와 일치하는지 확인한 후, 유효한 궤적만을 학습셋에 추가한다.

### 3. 정책 설계 (Policy Design)
-   **입력 및 출력**: 입력으로는 전면 및 손목 카메라의 RGB 이미지와 로봇의 고유 수용 감각(Proprioception) 상태를 사용한다. 출력으로는 6-DoF 엔드 이펙터 포즈와 그리퍼 동작을 예측한다.
-   **Diffusion Policy**: 1D temporal CNN 기반의 U-Net 구조를 사용한다.
-   **Action Chunking**: 한 번에 $T_p = 32$개의 미래 행동을 예측하고, 그중 $T_a = 8$개만 실행한 뒤 다시 관측값을 받아 행동을 업데이트한다. 이는 행동의 일관성을 유지하면서도 환경 변화에 반응하게 한다.
-   **손실 함수**: 표준적인 DDPM의 노이즈 예측 손실 함수를 사용한다.
$$L(\theta) = \mathbb{E}_{k, x_0, \epsilon} [\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_k}x_0 + \sqrt{1-\bar{\alpha}_k}\epsilon, k)\|^2]$$

### 4. Collect-and-Infer (CI)
모델 평가 단계에서 성공하거나 부분적으로 성공한 궤적을 저장하여 학습 데이터셋 $D_{H+E} = D_H \cup D_E$를 구성한다. 이를 통해 사람이 수집하지 못한 다양한 초기 상태에 대한 지도 데이터를 자동으로 확보한다.

## 📊 Results

### 1. 실험 설정
-   **데이터셋 및 작업**: FurnitureBench의 4가지 작업(Round table, Lamp, Square table, One leg)을 대상으로 하였다. 작업의 호라이즌은 최대 2,500 타임스텝에 달하며, 최대 5개의 부품을 조립해야 한다.
-   **기준선(Baselines)**: Action Chunking이 없는 MLP(MLP-NC), Chunking이 적용된 MLP(MLP-C), 기본 Diffusion Policy(DP-BC), 그리고 상태 노이즈 주입 방식($[9]$)을 비교 대상으로 하였다.
-   **평가 지표**: 100회의 롤아웃을 통해 최종 조립 성공률(Success Rate)의 평균 및 최대값을 측정하였다.

### 2. 주요 결과
-   **아키텍처 영향**: MLP-NC는 모든 작업에서 실패하였으나, Action Chunking을 적용한 MLP-C와 Diffusion Policy는 유의미한 성능 향상을 보였다. 특히 호라이즌이 긴 작업일수록 Diffusion Policy의 성능이 월등히 높았다.
-   **구성 요소별 기여**:
    -   **TA (Trajectory Augmentation)**: 정밀 삽입이 핵심인 `roundtable`과 `oneleg` 작업에서 성능을 크게 향상시켰다.
    -   **CI (Collect-and-Infer)**: `lamp` 작업(굴러가는 전구 파지)과 `squaretable` (긴 호라이즌) 작업에서 특히 효과적이었다.
    -   **TA & CI 결합**: 모든 작업에서 가장 높은 평균 성공률을 기록하였다 (예: `oneleg` 74%, `roundtable` 32%).
-   **데이터 효율성**: 단 10개의 시연 데이터만 사용하더라도 JUICER 파이프라인(TA + CI)을 적용하면 50개의 시연을 사용한 일반 BC보다 높은 성능(71% vs 59%)을 달성하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
본 연구는 데이터 효율적인 모방 학습을 위해 **'합성 데이터(TA)'**와 **'실행 데이터(CI)'**라는 두 가지 서로 다른 확장 전략이 상호보완적임을 입증하였다. TA는 정밀한 지점 주변의 밀도를 높여 Covariate Shift를 줄이는 반면, CI는 전체적인 상태 공간의 커버리지를 넓혀 다양한 초기 상황에 대응하게 한다. 또한, 다중 작업 학습(Multitask Learning) 시 서로 다른 조립 작업 간에 공통된 기술(Skill)이 전이되어 성능이 향상됨을 확인하였다.

### 2. 한계 및 미해결 과제
-   **시뮬레이션 의존성**: 모든 실험이 시뮬레이션 환경에서 수행되었으며, 실제 환경으로의 전이(Sim-to-Real)는 다뤄지지 않았다.
-   **리셋 가정**: TA와 CI 과정에서 시스템을 특정 상태로 되돌리는 '자동 리셋' 능력이 필수적이다. 이는 실제 환경에서 구현하기 매우 까다로운 제약 사항이다.
-   **일반화 능력**: 임의의 초기 부품 배치 분포에 대해 완전히 일반화된 정책을 생성하는 데에는 여전히 한계가 있다.

### 3. 비판적 해석
제안된 파이프라인은 매우 효과적이지만, 실질적으로는 '병목 상태'를 인간이 직접 지정해야 한다는 수동적인 단계가 포함되어 있다. 비록 이 시간이 짧다고 주장하지만, 작업의 복잡도가 증가함에 따라 병목 지점을 정확히 정의하는 것이 어려워질 수 있다. 향후에는 이러한 병목 지점을 자동으로 탐색하는 메커니즘이 추가될 필요가 있다.

## 📌 TL;DR

본 논문은 적은 양의 시연 데이터로 복잡한 로봇 조립 작업을 수행하기 위한 **JUICER 파이프라인**을 제안한다. **Diffusion Policy**를 기반으로 하며, 정밀 제어를 위한 **병목 상태 기반 궤적 증강(TA)**과 데이터 다양성을 위한 **반복적 롤아웃 수집(CI)**을 결합하였다. 실험 결과, 이 방식은 데이터 수집 비용을 획기적으로 줄이면서도 고정밀, long-horizon 조립 작업을 성공적으로 수행하며, 단 10개의 시연만으로도 높은 성능을 낼 수 있음을 보여주었다. 이는 향후 산업 현장에서 로봇에게 빠르게 새로운 조립 공정을 가르치는 기술로 응용될 가능성이 높다.