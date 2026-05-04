# Zero-shot Generative Model Adaptation via Image-specific Prompt Learning

Jiayi Guo, Chaofei Wang, You Wu, Eric Zhang, Kai Wang, Xingqian Xu, Shiji Song, Humphrey Shi, Gao Huang (2023)

## 🧩 Problem to Solve

본 논문은 사전 학습된 소스 도메인 생성기(Source-domain Generator)를 학습 데이터가 전혀 없는 새로운 타겟 도메인으로 적응시키는 **Zero-shot Generative Model Adaptation** 문제를 다룬다. 기존의 대표적인 방법론인 StyleGAN-NADA는 타겟 도메인의 텍스트 라벨만을 이용하여 효율적으로 적응을 수행하지만, 생성된 이미지의 품질이 제한적이고 **Mode Collapse**(생성 결과물이 특정 패턴으로 쏠리는 현상) 문제가 발생한다는 치명적인 한계가 있다.

이러한 Mode Collapse의 핵심 원인은 모든 이미지 쌍에 대해 동일한 **고정된 적응 방향(Fixed adaptation direction)**을 적용하기 때문이다. 수동으로 설계된 프롬프트(예: "A photo of a [domain]")를 사용하면 모든 이미지에 동일한 감독 신호가 전달되어, 이미지 개별의 특성이 무시되고 타겟 도메인의 전형적이지만 단조로운 패턴(예: 특정 도메인에서의 일관된 눈 모양이나 피부 톤)만 반복적으로 나타나게 된다. 따라서 본 논문의 목표는 이미지별 특성을 반영한 정밀한 적응 방향을 생성하여 이미지의 품질과 다양성을 높이고 Mode Collapse를 완화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모든 이미지에 공통으로 적용되는 고정 프롬프트 대신, 각 소스 이미지의 특성에 맞춤화된 **이미지별 프롬프트(Image-specific Prompt)**를 학습하는 **IPL(Image-specific Prompt Learning)** 방법론을 제안하는 것이다.

핵심 직관은 소스 이미지마다 서로 다른 프롬프트 벡터를 생성함으로써, 각 이미지 쌍에 대해 더욱 정밀하고 다양화된 적응 방향을 설정할 수 있다는 점이다. 이를 통해 타겟 도메인 생성기가 더 높은 유연성을 갖게 되어, 소스 이미지의 고유한 특징을 유지하면서도 타겟 도메인의 스타일을 정교하게 입힐 수 있게 된다.

## 📎 Related Works

### 생성 모델 적응 (Generative Model Adaptation)

기존 연구들은 타겟 도메인의 데이터 양에 따라 Few-shot과 Zero-shot으로 나뉜다. Few-shot 방식은 데이터 증강이나 정규화 기법을 통해 과적합을 방지하며 파인튜닝을 수행하지만, 여전히 타겟 데이터가 필요하고 적대적 학습(Adversarial training)의 높은 비용이 발생한다. 반면, NADA와 같은 Zero-shot 방식은 CLIP 모델을 도입하여 텍스트 라벨만으로 도메인 갭을 메우려 하지만, 앞서 언급한 Mode Collapse 문제가 발생한다.

### 프롬프트 학습 (Prompt Learning)

NLP 분야에서 시작된 프롬프트 학습은 수동 설계의 한계를 극복하기 위해 최적의 프롬프트 벡터를 자동으로 탐색하는 방식이다. 최근 CV 분야에서도 CoOp와 같은 연구가 진행되었으나, 이는 주로 도메인 전체에 대해 하나의 최적 프롬프트를 찾는 방식이다. 본 논문은 여기서 더 나아가 **개별 이미지 수준**에서 프롬프트를 동적으로 생성하는 latent mapper를 도입했다는 점에서 기존 프롬프트 학습과 차별화된다.

## 🛠️ Methodology

IPL은 크게 두 단계의 파이프라인으로 구성된다.

### Stage 1: 이미지별 프롬프트 학습 (Image-specific Prompt Learning)

이 단계에서는 소스 이미지의 잠재 코드(Latent code) $w_i$를 입력받아 이미지별 프롬프트 벡터 집합 $\{[V]^i_1, [V]^i_2, \dots, [V]^i_m\}$을 생성하는 **Latent Mapper $F$**를 학습시킨다.

1. **이미지별 프롬프트 행렬 구성**:
    이미지 $i$에 대한 프롬프트 행렬 $M^i_s$는 학습된 프롬프트 벡터들과 소스 도메인 라벨 $[Y_s]$의 임베딩을 결합하여 구성한다.
    $$M^i_s = F(w_i, \theta)[Y_s] = [[V]^i_1, [V]^i_2, \dots, [V]^i_m, [Y_s]]$$

2. **대조 학습 손실 함수 (Contrastive Learning Loss)**:
    학습된 프롬프트 벡터가 소스 이미지의 개별 특징을 잘 보존하도록 하기 위해, 이미지 임베딩과 프롬프트 행렬 임베딩 간의 코사인 유사도를 기반으로 대조 학습을 수행한다.
    $$\text{Sim}_{ij} = \text{Cos}(\text{Norm}(E^I(G_s(w_i))), \text{Norm}(E^T(M^j_s)))$$
    $$L_{contr} = \mathbb{E}_{w \in W} \left( \sum_{i \neq j} (\text{Sim}_{ij}) - \sum_{i=j} (\text{Sim}_{ij}) \right)$$

3. **도메인 정규화 손실 함수 (Domain Regularization Loss)**:
    학습된 프롬프트가 타겟 도메인과 충돌하지 않고 호환되도록 강제한다. 예를 들어 타겟이 '엘프'라면 '둥근 귀'라는 특징이 프롬프트에 포함되지 않도록 타겟 라벨 $Y_t$와의 유사도를 높인다.
    $$L_{domain} = -\mathbb{E}_{w_i \in W} \sum_{i=1}^n (\text{Cos}(E^T(M^i_t), E^T(Y_t)))$$

최종 손실 함수는 $L = L_{contr} + \lambda L_{domain}$으로 정의된다.

### Stage 2: 생성기 학습 (Generator Training)

Stage 1에서 학습된 $F$를 고정한 채, 타겟 생성기 $G_t$를 학습시킨다. 이때 개선된 **Directional CLIP Loss**를 사용한다.

1. **이미지 적응 방향 ($\Delta I_i$)**:
    소스 이미지와 타겟 이미지의 CLIP 이미지 공간에서의 차이를 계산한다.
    $$\Delta I_i = \text{Norm}(E^I(G_t(w_i))) - \text{Norm}(E^I(G_s(w_i)))$$

2. **텍스트 적응 방향 ($\Delta T_i$)**:
    학습된 Latent Mapper를 통해 생성된 이미지별 프롬프트를 사용하여 텍스트 공간에서의 적응 방향을 계산한다.
    $$\Delta T_i = \text{Norm}(E^T(M^i_t)) - \text{Norm}(E^T(M^i_s))$$

3. **최종 적응 손실 함수 ($L_{adapt}$)**:
    두 방향 벡터 사이의 각도를 최소화하여 이미지의 변화 방향이 텍스트가 지시하는 방향과 일치하도록 만든다.
    $$L_{adapt} = \mathbb{E}_{w_i \in W} \sum_{i=1}^n \left( 1 - \frac{\Delta I_i \cdot \Delta T_i}{|\Delta I_i| |\Delta T_i|} \right)$$

## 📊 Results

### 실험 설정

- **데이터셋**: FFHQ(인물), AFHQ(동물)
- **비교 대상**: NADA (Zero-shot GAN), DiffusionCLIP (Zero-shot Diffusion)
- **평가 지표**: Inception Score (IS, 품질 및 다양성), SIFID (타겟 스타일 일치도), SCS (구조 보존), ID (정체성 보존) 및 User Study.

### 주요 결과

1. **정량적 성능**: Table 1에 따르면 IPL은 거의 모든 설정에서 NADA보다 높은 IS를 기록하여 이미지 품질과 다양성이 크게 향상되었음을 입증했다. 특히 SIFID에서 우수한 성능을 보여 타겟 도메인의 스타일을 더 정확하게 구현했다.
2. **정성적 결과**: Figure 4에서 볼 수 있듯, NADA에서 나타나던 특징적인 Mode Collapse(예: 픽사 캐릭터의 우울한 표정, 애니메이션의 찡그린 눈)가 IPL에서는 사라지고 훨씬 다양하고 자연스러운 결과물이 생성되었다.
3. **사용자 평가**: 121명의 사용자가 참여한 연구에서 평균 80.5%가 IPL의 결과물을 선호하였다.
4. **범용성**: IPL은 GAN 기반의 StyleGANv2뿐만 아니라 Diffusion 모델(Diff-IPL)에도 적용 가능하며, Diffusion 모델과 결합했을 때 특히 복잡한 이미지에서 더 강력한 정체성 보존 능력을 보여주었다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구의 가장 큰 성과는 **"프롬프트를 이미지 단위로 동적화"** 함으로써 Zero-shot 적응의 고질적 문제인 Mode Collapse를 해결했다는 점이다. 이는 단순히 텍스트 라벨에 의존하는 것이 아니라, 소스 이미지의 잠재 공간 정보를 텍스트 임베딩 공간으로 매핑함으로써 생성기가 참고할 수 있는 감독 신호의 해상도를 높였기 때문이다.

### 한계 및 논의사항

1. **해석 가능성**: 학습된 프롬프트 벡터를 실제 단어와 매핑하려 시도했으나(Figure 14), 많은 경우 정확한 단어로 해석되지 않았다. 이는 프롬프트 벡터가 여러 복합적인 의미를 응축하고 있기 때문으로 추측되며, 이에 대한 해석 가능성(Interpretability) 확보가 향후 과제로 남는다.
2. **큰 도메인 변화**: 인물에서 고양이로 변환하는 것과 같이 도메인 간 거리가 매우 먼 경우(Large Domain Shift)에는 여전히 성능 한계가 존재하며, 이는 추가적인 연구가 필요함을 시사한다.
3. **하이퍼파라미터 $\lambda$**: 도메인 정규화 계수 $\lambda$가 너무 작으면 소스 특징이 과하게 보존되어 타겟 스타일이 부족해지고, 너무 크면 다시 Mode Collapse와 유사한 패턴이 나타나는 트레이드-오프 관계가 관찰되었다.

## 📌 TL;DR

본 논문은 Zero-shot 생성 모델 적응에서 발생하는 Mode Collapse 문제를 해결하기 위해, 이미지마다 서로 다른 최적의 프롬프트를 생성하는 **Image-specific Prompt Learning (IPL)**을 제안한다. Latent Mapper를 통해 이미지별 맞춤형 적응 방향을 설정함으로써 생성 결과물의 다양성과 품질을 획기적으로 높였으며, 이는 GAN과 Diffusion 모델 모두에 적용 가능한 범용적인 프레임워크이다. 향후 이 기술은 데이터가 부족한 예술적 도메인의 이미지 합성 및 데이터 증강 분야에서 핵심적인 역할을 할 것으로 기대된다.
