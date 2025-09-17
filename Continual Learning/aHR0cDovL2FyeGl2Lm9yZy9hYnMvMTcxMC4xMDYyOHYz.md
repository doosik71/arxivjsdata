# VARIATIONAL CONTINUAL LEARNING

Cuong V. Nguyen, Yingzhen Li, Thang D. Bui, Richard E. Turner

## 🧩 Problem to Solve

지속 학습(continual learning)은 데이터가 순차적으로 도착하고, 태스크가 시간에 따라 변화하며, 새로운 태스크가 계속해서 출현하는 온라인 학습의 한 형태로, 기계 학습의 핵심 과제 중 하나입니다. 심층 신경망은 새로운 데이터에 적응하면서도 이전 태스크에서 학습한 지식을 유지하는 데 어려움을 겪으며, 이는 흔히 **치명적인 망각(catastrophic forgetting)** 문제로 이어집니다. 기존의 지속 학습 방법들은 안정성(stability)과 유연성(plasticity) 사이의 균형을 맞추는 데 한계가 있거나, 수동적인 하이퍼파라미터 튜닝을 필요로 합니다.

## ✨ Key Contributions

* **변분 지속 학습(Variational Continual Learning, VCL) 프레임워크 제안**: 온라인 변분 추론(Variational Inference, VI)과 신경망을 위한 몬테카를로 VI 기법을 결합하여 치명적인 망각을 피하고 자동적으로 지식을 유지하는 일반적인 지속 학습 방법론을 개발했습니다.
* **다목적 적용 가능성**: 제안된 VCL 프레임워크가 심층 판별 모델(deep discriminative models)과 심층 생성 모델(deep generative models) 모두에 성공적으로 적용될 수 있음을 입증했습니다.
* **소규모 에피소드 메모리(Episodic Memory) 통합**: Coreset 데이터 요약 방법을 VI와 결합하여 소규모 에피소드 메모리를 VCL에 포함시켰으며, 이는 이전 태스크의 핵심 정보를 유지하여 망각을 완화하는 데 기여합니다.
* **최첨단 성능 달성**: 다양한 지속 학습 벤치마크 태스크에서 기존의 최첨단 방법(EWC, SI, LP)을 능가하는 성능을 보였으며, VCL은 하이퍼파라미터 튜닝 없이도 우수한 결과를 얻었습니다.

## 📎 Related Works

* **심층 판별 모델을 위한 지속 학습**:
  * **정규화된 최대 우도 추정(Regularized Maximum Likelihood Estimation)**: $\mathcal{L}_t(\theta) = \sum_{n=1}^{N_t} \text{log}p(y^{(n)}_t|\theta,x^{(n)}_t) - \frac{1}{2}\lambda_t(\theta-\theta_{t-1})^{\text{T}}\Sigma^{-1}_{t-1}(\theta-\theta_{t-1})$ 형태의 목적 함수를 최적화합니다.
    * **최대 우도 추정(MLE) 및 MAP 추정**: 정규화 항이 없거나(MLE), 가우시안 사전 분포로 해석하여 MAP 추정을 수행하지만, $\Sigma_t$ 계산이 어렵거나 망각에 취약합니다.
    * **라플라스 전파(Laplace Propagation, LP)**: 각 단계에서 라플라스 근사를 적용하여 $\Sigma^{-1}_t$를 재귀적으로 업데이트합니다.
    * **탄력적 가중치 고정(Elastic Weight Consolidation, EWC)**: 피셔 정보 행렬(Fisher Information Matrix)을 사용하여 이전 태스크에 중요한 파라미터를 보호함으로써 망각을 완화합니다.
    * **시냅스 지능(Synaptic Intelligence, SI)**: 각 파라미터가 태스크에 미치는 중요도를 측정하여 정규화에 활용합니다.
  * **베이즈 신경망 학습**: 확장 칼만 필터링, 라플라스 근사, 변분 추론(VI), 순차 몬테카를로, 기댓값 전파(EP) 등이 있으나, 대부분 배치 학습에 중점을 두었습니다.
  * **온라인 변분 추론**: 이전에 연구되었으나, 신경망이나 복잡한 관련 태스크에는 적용되지 않았습니다.
* **심층 생성 모델을 위한 지속 학습**:
  * **VAE (Variational Auto-Encoders)**: 일반적인 VAE 알고리즘을 새로운 데이터셋에 직접 적용하면 치명적인 망각이 발생합니다.
  * **EWC 정규화가 추가된 VAE**: EWC 정규화 항을 VAE 목적 함수에 추가하지만, 피셔 정보 행렬 계산에 어려움이 있습니다.

## 🛠️ Methodology

VCL은 근사 베이즈 추론을 통해 지속 학습을 수행합니다.

1. **베이즈 추론의 원리**:
    * Bayes' rule에 따라 현재 데이터 $D_T$를 관찰한 후의 사후 분포 $p(\theta|D_{1:T})$는 이전 사후 분포 $p(\theta|D_{1:T-1})$와 현재 데이터의 우도 $p(D_T|\theta)$를 곱하여 재정규화함으로써 얻어집니다. 이는 온라인 업데이트를 자연스럽게 지원합니다.
    * $$p(\theta|D_{1:T}) \propto p(\theta|D_{1:T-1})p(D_T|\theta)$$
2. **변분 지속 학습(VCL)**:
    * 정확한 베이즈 추론은 대부분 다루기 어렵기 때문에, VCL은 KL 발산 최소화를 통해 근사 사후 분포 $q_t(\theta)$를 찾습니다.
    * $$q_t(\theta) = \text{arg min}_{q \in Q} \text{KL} \left( q(\theta) \middle\| \frac{1}{Z_t} q_{t-1}(\theta)p(D_t|\theta) \right)$$
    * 여기서 $q_0(\theta)$는 사전 분포 $p(\theta)$로 정의됩니다.
3. **에피소드 메모리(Coreset) 강화 VCL (Algorithm 1)**:
    * 반복적인 근사로 인한 오류 누적 및 망각을 완화하기 위해 소규모 코어셋 $C_t$ (이전 태스크의 대표 데이터 포인트)를 유지합니다.
    * **코어셋 업데이트**: 현재 태스크의 새로운 데이터 포인트와 이전 코어셋 $C_{t-1}$에서 선택된 데이터 포인트로 $C_t$를 구성합니다 (무작위 샘플링 또는 K-중심(K-center) 알고리즘 사용).
    * **변분 분포 업데이트**:
        * 코어셋에 포함되지 않은 데이터를 사용하여 중간 변분 분포 $\tilde{q}_t(\theta)$를 업데이트합니다.
        * $$\tilde{q}_t(\theta) \leftarrow \text{arg min}_{q \in Q} \text{KL} \left( q(\theta) \middle\| \frac{1}{\tilde{Z}} \tilde{q}_{t-1}(\theta)p(D_t \cup C_{t-1} \setminus C_t |\theta) \right)$$
        * 최종 변분 분포 $q_t(\theta)$는 예측에만 사용되며, $\tilde{q}_t(\theta)$와 코어셋 $C_t$의 우도를 결합하여 업데이트합니다.
        * $$q_t(\theta) \leftarrow \text{arg min}_{q \in Q} \text{KL} \left( q(\theta) \middle\| \frac{1}{Z} \tilde{q}_t(\theta)p(C_t|\theta) \right)$$
4. **심층 판별 모델 적용**:
    * **네트워크 아키텍처**: Multi-head 네트워크를 사용합니다. 입력에 가까운 파라미터($\theta_S$)는 공유하고, 각 태스크($t$)에 대한 출력 레이어($\theta_{H_t}$)는 분리합니다.
    * **근사 사후 분포**: 가우시안 평균 필드(Gaussian mean-field) 근사를 사용합니다: $q_t(\theta) = \prod_{d=1}^D \mathcal{N}(\theta_{t,d}; \mu_{t,d}, \sigma^2_{t,d})$.
    * **학습 목적**: 온라인 변분 자유 에너지(online variational free energy)의 음수 또는 온라인 주변 우도(online marginal likelihood)의 변분 하한을 최대화합니다.
    * $$\mathcal{L}^{\text{VCL}}_t(q_t(\theta)) = \sum_{n=1}^{N_t} \mathbb{E}_{\theta \sim q_t(\theta)} [\text{log}p(y^{(n)}_t|\theta,x^{(n)}_t)] - \text{KL}(q_t(\theta)||q_{t-1}(\theta))$$
    * 기울기 계산에는 몬테카를로 근사와 국소 재매개변수화 트릭(local reparameterization trick)을 사용합니다.
5. **심층 생성 모델 적용 (VAE)**:
    * 모델 파라미터 $\theta$와 인코더(변분 파라미터 $\phi$)에 대한 전체 변분 하한(full variational lower bound)을 최대화하여 사후 분포 $q_t(\theta)$를 근사합니다.
    * $$\mathcal{L}^{\text{VCL}}_t(q_t(\theta),\phi) = \mathbb{E}_{q_t(\theta)} \left\{ \sum_{n=1}^{N_t} \mathbb{E}_{q_\phi(z^{(n)}_t|x^{(n)}_t)} \left[ \text{log}\frac{p(x^{(n)}_t|z^{(n)}_t,\theta)p(z^{(n)}_t)}{q_\phi(z^{(n)}_t|x^{(n)}_t)} \right] \right\} - \text{KL}(q_t(\theta)||q_{t-1}(\theta))$$
    * 생성 모델도 공유 및 태스크별 컴포넌트로 구성될 수 있습니다.

## 📊 Results

VCL은 세 가지 판별 태스크와 두 가지 생성 태스크에서 평가되었습니다. 경쟁 모델(EWC, SI, LP)은 하이퍼파라미터 튜닝을 거쳤음에도 불구하고 VCL이 더 좋은 성능을 보였습니다.

* **심층 판별 모델 (Permuted MNIST, Split MNIST, Split notMNIST)**:
  * **Permuted MNIST**: 10개 태스크 후 평균 정확도에서 VCL (90%)은 EWC (84%), SI (86%), LP (82%)를 크게 앞섰습니다. Coreset을 VCL과 결합하면 (무작위 Coreset 및 K-center Coreset 모두) 93%의 정확도로 성능이 더욱 향상되었습니다. Coreset 크기가 커질수록 성능은 더 향상되어, 5,000개 예제 Coreset으로 95.5%에 도달했습니다.
  * **Split MNIST**: VCL (97.0%)은 EWC (63.1%)와 LP (61.2%)를 크게 능가했으며, SI (98.9%)와는 근소한 차이로 뒤처졌습니다. Coreset 추가 시 VCL은 약 98.4% 정확도로 SI에 매우 근접했습니다.
  * **Split notMNIST**: VCL (92.0%)은 EWC (71%)와 LP (63%)보다 훨씬 우수했으며, SI (94%)와 경쟁적인 성능을 보였습니다. 무작위 Coreset 추가 시 VCL은 96%의 정확도로 성능이 향상되었습니다.
* **심층 생성 모델 (MNIST 숫자 생성, notMNIST 문자 생성)**:
  * 순진한(naive) 온라인 학습 방식은 치명적인 망각을 겪는 반면, LP, EWC, SI, VCL은 이전 태스크를 기억했습니다.
  * 생성된 이미지의 시각적 품질 측면에서 SI와 VCL이 가장 우수했습니다.
  * 정량적 평가(테스트 로그-우도 및 분류기 불확실성)에서 VCL은 SI와 동등하거나 약간 더 나은 성능을 보였으며, LP와 EWC는 더 낮은 성능을 나타냈습니다.
  * VCL은 목적 함수에 튜닝된 하이퍼파라미터가 없음에도 불구하고, 모든 지표에서 전반적으로 우수한 장기 기억 성능을 입증했습니다.

## 🧠 Insights & Discussion

* **베이즈 추론의 적합성**: 베이즈 추론은 파라미터에 대한 분포를 유지하고, 새로운 데이터가 도착할 때 이전 사후 분포를 사전 분포로 사용하여 지식을 자연스럽게 통합함으로써 지속 학습을 위한 강력하고 원칙적인 프레임워크를 제공합니다. 이는 치명적인 망각을 본질적으로 완화합니다.
* **VCL의 장점**: VCL은 완전한 파라미터 분포를 유지하여 불확실성 추정치를 제공하며, 이는 기존 지식의 중요도를 판단하고 새로운 학습에 적절히 반영하는 데 중요합니다. 또한, MAP, EWC, SI와 달리 검증 세트에서 튜닝해야 하는 자유 하이퍼파라미터가 없어 온라인 학습 환경에서 유리하고, 더욱 자동적인 학습을 가능하게 합니다.
* **에피소드 메모리의 효과**: 코어셋과 같은 에피소드 메모리는 반복적인 근사로 인한 정보 손실을 보완하고 성능을 더욱 향상시키는 효과적인 방법입니다. 코어셋 단독으로는 성능이 좋지 않더라도, VCL과 결합될 때 시너지 효과를 냅니다.
* **한계 및 향후 연구**: 현재 VCL은 모델 구조가 사전에 알려져 있다는 가정을 하지만, 향후에는 새로운 태스크에 따라 모델 구조를 자동으로 구축하는 연구가 필요합니다. 또한, 더 정교한 에피소드 메모리 기법과 다른 근사 추론 방법들을 탐색할 수 있습니다. VCL은 순차적 의사 결정 문제(예: 강화 학습, 능동 학습)에서 효율적인 모델 개선에 특히 적합합니다.

## 📌 TL;DR

VCL은 온라인 변분 추론을 활용하여 신경망의 치명적인 망각 문제를 해결하는 지속 학습 프레임워크입니다. 이 방법은 베이즈 추론의 원리를 기반으로 파라미터 분포를 유지하고, 코어셋 기반 에피소드 메모리를 통해 성능을 강화합니다. 판별 및 생성 모델 모두에서 기존 최첨단 방법들을 능가하는 성능을 보였으며, 하이퍼파라미터 튜닝 없이도 강력하고 자동적인 학습을 가능하게 합니다.
