# A Comprehensive Empirical Evaluation on Online Continual Learning

Albin Soutif–Cormerais, Antonio Carta, Andrea Cossu, Julio Hurtado, Hamed Hemati, Vincenzo Lomonaco, Joost van de Weijer (2023)

## 🧩 Problem to Solve

본 논문은 Online Continual Learning (OCL) 환경에서 다양한 학습 방법론들이 실제로 어느 정도의 성능을 보이는지를 정밀하게 분석하고 평가하는 것을 목표로 한다.

일반적인 Continual Learning은 데이터가 태스크 단위의 배치(batch) 형태로 제공되며, 태스크 경계가 명확하다는 가정을 가진다. 반면, OCL은 실제 현실 세계의 시나리오와 더 유사하게, 데이터가 스트림(stream) 형태로 유입되며 모델이 각 데이터 포인트에 대해 실시간으로 학습해야 한다. 이 과정에서 OCL은 다음과 같은 핵심적인 문제들을 해결해야 한다.

첫째, 데이터 분포가 시간에 따라 변하는 Non-stationary한 특성으로 인해, 새로운 지식을 습득하는 Plasticity(가소성)와 기존 지식을 유지하는 Stability(안정성) 사이의 균형을 맞추는 Stability-Plasticity Dilemma를 해결해야 한다. 둘째, 메모리와 계산 자원이 매우 제한적이므로 최소한의 데이터 저장만으로 망각(Forgetting)을 방지해야 한다. 셋째, 모델이 학습 도중 언제라도 추론에 사용될 수 있어야 하는 Anytime Inference 능력을 갖추어야 한다.

결과적으로 본 연구는 기존 OCL 연구들이 주로 최종 정확도와 망각률에만 집중했다는 한계를 지적하며, 안정성, 표현 학습의 질(Representation Quality) 등 다각적인 지표를 통해 OCL 방법론들을 종합적으로 평가하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 OCL에 대한 엄격한 정의와 다차원적인 평가 체계를 구축하고, 이를 통해 기존 방법론들의 실질적인 효용성을 검증한 점에 있다. 구체적인 기여 사항은 다음과 같다.

1. **OCL의 정식 정의 및 평가 지표 제안**: Accuracy뿐만 아니라 Worst-Case Accuracy(WC-ACC), Average Anytime Accuracy(AAA), Cumulative Forgetting, 그리고 Probed Accuracy와 같은 지표를 통해 안정성과 표현 학습의 질을 측정하는 프레임워크를 정의하였다.
2. **광범위한 실증적 분석**: Split-CIFAR100 및 Split-TinyImagenet 벤치마크를 사용하여 9가지의 다양한 OCL 접근 방식을 비교 분석하였다.
3. **핵심 인사이트 도출**: OCL 모델들이 겪는 성능 저하의 주원인이 특징 추출기(Backbone)의 표현력 부족이 아니라, 분류기(Classifier)의 학습 문제와 Underfitting에 있음을 밝혀냈다.
4. **오픈소스 코드 제공**: Avalanche 프레임워크를 기반으로 구현된 모듈화된 코드베이스를 공개하여 향후 연구의 재현성과 확장성을 높였다.

## 📎 Related Works

기존의 Continual Learning 관련 서베이들은 주로 Task-Incremental Learning이나 Class-Incremental Learning의 오프라인 설정(데이터셋 전체에 접근 가능한 경우)에 집중해 왔다. 최근 OCL에 대한 실증적 평가 연구(예: Mai et al. [32])가 등장하였으나, 본 논문은 다음과 같은 차별점을 갖는다.

먼저, 단순한 성능 비교를 넘어 안정성(Stability)과 잠재 표현의 품질(Representation Quality)을 측정하는 더 정교한 지표를 도입하였다. 또한, 단순히 최신 기법들을 나열하는 것이 아니라, Rehearsal-based 방법론들의 구성 요소(Sampling, Loss, Classifier, Weight Update)를 체계적으로 분류하고 각 요소가 성능에 미치는 영향을 분석하였다. 특히, 가장 기본적인 Experience Replay(ER)가 적절히 튜닝되었을 때 얼마나 강력한 베이스라인이 될 수 있는지를 보여줌으로써, 복잡한 알고리즘 설계보다 하이퍼파라미터 튜닝과 구현 디테일의 중요성을 강조한다.

## 🛠️ Methodology

### 전체 시스템 구조

본 논문에서 평가하는 대부분의 방법론은 Rehearsal-based 접근 방식을 따른다. 이는 고정된 크기의 메모리 $M$을 유지하며, 새로운 데이터가 유입될 때마다 메모리에서 과거 샘플을 추출하여 함께 학습하는 구조이다.

OCL 알고리즘은 다음과 같은 함수로 정의된다:
$$A: (x_t, y_t), f_{t-1}, M_{t-1} \to f_t, M_t$$
여기서 $f_t$는 시간 $t$에서의 모델, $M_t$는 과거 샘플들의 집합인 리플레이 버퍼를 의미한다.

### 주요 구성 요소 및 방법론

분석 대상이 된 방법론들은 기본 ER 구조에서 다음과 같은 요소들을 수정하여 구현되었다.

1. **Sampling (샘플링)**:
   - **MIR**: 새로운 데이터 학습 후 손실 함수가 가장 많이 증가한(Maximal Interfered) 샘플을 선택하여 리플레이한다.
   - **RAR**: 결정 경계 근처의 샘플을 생성하기 위해 Adversarial Attack을 이용한 증강 샘플을 사용한다.
2. **Loss (손실 함수)**:
   - **ER-ACE**: 새로운 데이터와 메모리 데이터에 대해 서로 다른 손실 함수를 적용한다.
   - **DER++**: Logits Distillation을 사용하여 과거 모델의 출력을 보존한다.
   - **SCR**: Supervised Contrastive Loss를 사용하여 클래스 간 변별력을 높인다.
3. **Classifier (분류기)**:
   - 일반적인 Linear Classifier 외에도, 각 클래스의 평균 벡터를 계산하여 거리를 측정하는 NCM(Nearest-Class-Mean) 분류기를 사용한다(예: SCR).
4. **Model Update (가중치 업데이트)**:
   - **A-GEM**: 메모리 데이터의 그래디언트를 제약 조건으로 사용하여, 새로운 학습이 과거의 지식을 파괴하지 않도록 그래디언트 투영(Projection)을 수행한다.

### 평가 지표 (Metrics)

모델의 다각적인 성능을 측정하기 위해 다음과 같은 수식들이 사용되었다.

- **Worst-Case Accuracy (WC-ACC)**: 현재 단계의 정확도와 과거 모든 태스크에서 겪은 최소 정확도의 가중 합으로, 모델의 최악의 상황에서의 안정성을 측정한다.
  $$WC\text{-}ACC_t = \frac{1}{k} A(E_k, f_t) + \left(1 - \frac{1}{k}\right) \min\text{-}ACC_{T_k}$$
- **Average Anytime Accuracy (AAA)**: 모든 학습 스텝에서 평가한 정확도의 평균으로, 학습 과정 전반의 성능을 나타낸다.
  $$AAA_t = \frac{1}{t} \sum_{j=1}^{t} \frac{1}{k} \sum_{i=1}^{k} A(E_i, f_j)$$
- **Cumulative Forgetting**: Class-Incremental 설정에 맞춰, 현재까지 학습한 클래스들만을 대상으로 한 정확도의 하락폭을 측정한다.
- **Probed Accuracy**: 학습된 백본(Backbone)을 동결시키고, 전체 데이터를 사용하여 선형 분류기를 다시 학습시켜 특징 표현(Representation) 자체의 품질을 측정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Split-CIFAR100 (20 태스크), Split-TinyImagenet (20 태스크)
- **모델**: Slim ResNet18, SGD Optimizer 사용
- **메모리 크기**: CIFAR100은 2,000개, TinyImagenet은 4,000개 샘플 저장
- **비교 대상**: ER, AGEM, ER+LwF, ER-ACE, MIR, SCR, RAR, DER++, GDumb 및 i.i.d. 기준 모델

### 주요 결과

1. **최종 정확도 (Final Accuracy)**:
   - 대부분의 방법론이 vanilla ER과 매우 유사한 성능을 보였다.
   - ER+LwF가 소폭 우세했으나, 전반적으로 i.i.d. 학습 결과와 약 5% 내외의 차이를 보였다.
   - AGEM은 다른 방법론들에 비해 현저히 낮은 성능을 기록하였다.

2. **안정성 (Stability)**:
   - 최종 정확도가 높다고 해서 반드시 안정성이 높은 것은 아니었다.
   - 예를 들어, SCR은 MIR보다 최종 정확도는 비슷하거나 낮았지만, WC-ACC 지표에서는 훨씬 높은 안정성을 보였다. 이는 SCR이 태스크 전환 시 발생하는 성능 급락을 더 잘 억제함을 의미한다.

3. **표현 학습의 질 (Representation Quality)**:
   - **가장 놀라운 발견**은 Probed Accuracy가 i.i.d. 학습 결과와 매우 유사하게 나타났다는 점이다. 이는 OCL 방법론들이 백본을 통해 유의미한 특징을 추출하는 능력은 충분히 갖추었음을 시사한다.

4. **망각과 역전이 (Forgetting & Backward Transfer)**:
   - Cumulative Forgetting 분석 결과, 많은 방법론에서 역전이(Backward Transfer)가 관찰되었다. 즉, 이후 태스크를 학습하면서 이전 태스크의 성능이 오히려 올라가는 현상이 발생했다.

## 🧠 Insights & Discussion

### 분석 및 비판적 해석

본 논문의 결과는 OCL 분야에 몇 가지 중요한 시사점을 던진다.

첫째, **분류기(Classifier)가 병목 지점이다**. Probed Accuracy가 i.i.d. 수준으로 높음에도 불구하고 실제 정확도가 낮은 것은, 백본의 특징 추출 능력이 부족해서가 아니라, 증분적으로 추가되는 클래스들에 대해 분류기가 적절한 결정 경계를 학습하지 못하고 있기 때문이다.

둘째, **Underfitting의 역설**이다. OCL 모델에서 역전이가 발생하는 이유는 모델이 충분히 학습되지 않은 Underfitted 상태이기 때문이다. 데이터 스트림의 특성상 각 샘플에 대해 학습 횟수가 매우 적기 때문에, 이후 태스크를 학습하는 과정이 결과적으로 이전 태스크에 대한 추가 학습 효과를 내어 성능이 상승하는 것으로 해석된다.

셋째, **구현 디테일의 지배력**이다. SCR의 성능 우위가 알고리즘 자체보다 '더 큰 메모리 배치 사이즈'를 사용했다는 점에 기인한다는 사실을 밝혀냈다. ER에 SCR과 동일한 메모리 배치 사이즈를 적용했을 때 성능이 비약적으로 상승한 점은, 많은 OCL 연구들이 하이퍼파라미터의 영향을 간과하고 알고리즘의 우수성으로 오인했을 가능성을 제기한다.

### 한계 및 미해결 질문

논문은 특정 하이퍼파라미터 튜닝 방식(초반 4개 태스크 사용)을 채택하였으나, 이것이 모든 시나리오에서 최적의 튜닝 방법인지에 대해서는 명시되지 않았다. 또한, 다양한 모델 아키텍처(예: ViT)에서의 검증이 부족하며, ResNet18이라는 단일 모델에 의존한 한계가 있다.

## 📌 TL;DR

본 논문은 OCL의 다양한 방법론들을 정밀한 지표(안정성, 표현 품질 등)로 분석한 종합 보고서이다. 실험 결과, **복잡한 최신 기법들이 기본적인 Experience Replay(ER)보다 압도적으로 뛰어나지 않으며, 특히 적절한 하이퍼파라미터 튜닝을 거친 ER은 매우 강력한 베이스라인임**을 입증하였다. 또한, OCL의 핵심 난제는 특징 추출 능력이 아니라 **분류기 학습과 Underfitting 문제**에 있음을 밝혀냈으며, 이는 향후 연구가 복잡한 아키텍처 설계보다는 효율적인 분류기 학습 및 튜닝 전략에 집중해야 함을 시사한다.
