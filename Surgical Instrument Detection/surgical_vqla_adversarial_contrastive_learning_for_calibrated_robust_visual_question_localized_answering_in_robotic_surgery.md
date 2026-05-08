# Surgical-VQLA++: Adversarial Contrastive Learning for Calibrated Robust Visual Question-Localized Answering in Robotic Surgery

Long Bai, Guankun Wang, Mobarakol Islam, Lalithkumar Seenivasan, An Wang, Hongliang Ren

## 🧩 Problem to Solve

기존 Surgical Visual Question Answering (VQA) 모델은 수술 장면의 시각적 정보와 임상적 의사결정 사이의 간극을 좁히는 데 기여했지만, 질문에 해당하는 관심 영역을 시각적으로 정확히 지목하지 못하여 수술 장면에 대한 이해가 불완전했습니다. 이에 따라 본 논문은 질문에 대한 답변과 함께 관련 영상 영역을 정확히 찾아주는 Surgical Visual Question Localized-Answering (VQLA)의 필요성을 제기합니다. 또한, 수술 환경의 안전성과 이미지 획득 및 전송 과정에서 발생할 수 있는 노이즈나 손상(corruption)으로 인해 VQLA 모델의 **견고성(robustness)**이 크게 저하될 수 있다는 문제점을 해결하고자 합니다.

## ✨ Key Contributions

* **Surgical-VQLA++ 프레임워크 제안**: 답변과 로컬라이제이션(localization) 간의 인스턴스 수준 연결을 구축하여 수술 VQLA의 성능과 견고성을 크게 향상시켰습니다. End-to-end 솔루션으로 150.6 FPS의 효율적인 추론 속도를 제공합니다.
* **Calibrated Co-attention Gated Vision-Language (C$^2$G-ViL) 임베딩 모듈 개발**: 멀티모달(multimodal) 표현의 정렬 및 상호작용을 강화하고, 전역 컨텍스트 특징을 보정하기 위해 제안되었습니다. 이는 최적의 융합 가중치를 탐색하고 모달리티 간의 정렬을 촉진합니다.
* **적대적 대조 학습(Adversarial Contrastive Learning) 전략 도입**: DeiT 백본을 기반으로 딥 특징 학습에 적대적 예제를 활용한 대조 학습 전략을 적용하여 모델의 성능과 견고성을 향상시켰습니다. 멀티태스크 수렴을 위해 손실 함수의 조합도 최적화했습니다.
* **확장된 Surgical VQLA 데이터셋 공개**: EndoVis-18-VQLA 및 EndoVis-17-VQLA 데이터셋을 확장하여 수술 도구에 대한 추가 질문(QA 쌍 17,269개)을 포함시켰으며, 수술 기관, 도구, 행동, 도구 위치 등 다양한 콘텐츠를 포괄합니다. 각 QA 쌍에는 답변에 해당하는 바운딩 박스(bounding box)가 포함됩니다.

## 📎 Related Works

* **컴퓨터 비전 분야의 Grounded VQA**: Antol et al. [22]의 VQA 태스크 도입 이후, 멀티모달 융합 [8, 9] 및 어텐션 메커니즘 [10]을 활용한 모델들이 발전했습니다. 시각적 그라운딩(visual grounding) [26, 27, 28]은 VQA의 해석 가능성을 높이는 중요한 연구 분야로 부상했으며, Fukui et al. [31]의 압축 빌리니어 풀링, Zhu et al. [32]의 듀얼 디코더 트랜스포머 등이 제안되었습니다. 그러나 이러한 접근 방식은 자연 이미지에 특화되어 추론 속도와 전역 컨텍스트 이해에 한계가 있었습니다.
* **의료 분야의 Grounded VQA**: 의료 VQA는 임상 진단 및 치료를 돕는 데 기여했지만 [36, 37, 38], 복잡한 해부학적 구조와 도메인 특이적 용어 등 고유한 과제를 안고 있습니다. Liu et al. [36]은 대조 사전 학습, Naseem et al. [37]은 지식 인식 멀티모달 표현, Liu et al. [13]은 답변 질의 디코더를 제안했습니다. 의료 분야에서 답변 그라운딩 또는 VQLA는 상대적으로 덜 탐구되었으며, Tascon et al. [45, 46] 및 Vu et al. [47]은 어텐션 맵을 사용하여 질문 영역을 지역화했지만, 멀티태스크 학습 프레임워크는 고려하지 않았습니다.

## 🛠️ Methodology

본 논문의 VQLA 시스템 아키텍처는 시각적 및 텍스트 표현을 처리하는 두 개의 특징 추출기(`C$^2$G-ViL` 임베딩 모듈 및 사전 학습된 DeiT 백본)와 분류 및 바운딩 박스 생성을 위한 두 개의 예측 헤드로 구성됩니다. 모델 성능 최적화를 위해 적대적 대조 학습 전략을 사용합니다.

1. **특징 추출 (Feature Extraction)**
    * **시각 특징 추출**: Faster RCNN [52] 대신 ImageNet [57]으로 사전 학습된 ResNet18 [56]을 사용하여 전역적인 특징을 추출합니다. 이는 수술 도구와 주변 환경을 포함한 전체적인 이해를 가능하게 하며, 더 빠른 추론 속도와 end-to-end 솔루션을 제공합니다.
    * **텍스트 특징 추출**: 수술 데이터셋에 특화된 토크나이저를 사용하여 질문을 임베딩 행렬로 변환합니다.
    * 두 모달리티의 특징은 `C$^2$G-ViL` 임베딩 모듈로 전달됩니다.

2. **C$^2$G-ViL 임베딩 (Calibrated Co-attention Gated Vision-Language Embedding)**
    * **Co-Attention Cross-Model Interaction**: 가이드-어텐션과 셀프-어텐션 메커니즘을 통합하여 모달리티 간의 상호작용을 강화하고, 텍스트 입력에 기반하여 시각 입력이 핵심 영역에 집중하도록 합니다. 6개의 co-attention 레이어를 사용합니다.
    * **Multimodal Collaborated Calibration**: 시각 임베딩을 텍스트 임베딩으로, 텍스트 임베딩을 시각 임베딩으로 보정하여 서로 다른 모달리티 간의 표현을 정렬하고 신뢰할 수 있는 대응 관계를 구축합니다.
    * **Global Contextual Calibration**: 페어와이즈 빌리니어 풀링(pairwise bilinear pooling) 기법을 사용하여 각 모달리티 내에서 전역 컨텍스트 의미론을 정제하고 강화합니다.
    * **Gated Fusion**: 게이팅 메커니즘을 도입하여 시각 및 텍스트 임베딩 입력에 대한 조합 가중치를 동적으로 조절하고, 멀티모달 표현의 효과적인 통합을 학습합니다.

3. **적대적 대조 학습 (Contrastive Training with Adversarial Examples)**
    * 텍스트 및 시각 임베딩에 적대적 섭동(perturbation) $r_t, r_v$를 개별적으로 적용합니다:
        $$r = -\epsilon \text{sign}(\nabla_x \mathcal{L}(F(x), G))$$
        $r_t = -\alpha \cdot \epsilon \text{sign}(\nabla_{x_t} \mathcal{L}(F(x_t), G))$
        $r_v = -\beta \cdot \epsilon \text{sign}(\nabla_{x_v} \mathcal{L}(F(x_v), G))$
        여기서 $\alpha, \beta$는 각 모달리티의 섭동 가중치입니다.
    * DeiT-base [18] 백본을 통해 정제된(clean) 임베딩과 섭동된(perturbed) 임베딩을 모두 학습합니다.
    * 대조 손실 $\mathcal{L}_{CTR}$을 사용하여 클래스 내 유사성과 클래스 간 차이점을 모두 포괄하는 견고한 표현 학습을 촉진합니다.
        $$\mathcal{L}_{CTR}(x; r) = -\log \frac{\exp(\text{cos}(P_i, P^r_j) / C)}{\sum_{2B, \gamma=1}^{0|\gamma=i} \exp(\text{cos}(P_i, P^r_\gamma) / C)}$$

4. **예측 헤드 및 손실 함수 (Prediction Heads and Loss Functions)**
    * 분류를 위한 선형 레이어와 바운딩 박스 회귀를 위한 3계층 FFN으로 구성됩니다.
    * 클래스 불균형 문제를 해결하기 위해 Focal Loss [66]를 QA 손실로 사용하고, GIoU Loss [65]를 로컬라이제이션 손실로 사용합니다.
    * 멀티태스크 학습에서 손실 간의 균형을 위해 불확실성 손실(uncertainty loss) [68]을 활용하여 각 태스크의 불확실성에 따라 동적으로 가중치를 할당합니다.
        $$\mathcal{L}_{VQLA}(x; G) = \frac{1}{2\sigma_1^2} \mathcal{L}_{Focal} + \log \sigma_1 + \frac{1}{2\sigma_2^2} \mathcal{L}_{GIoU} + \log \sigma_2$$
    * 최종 손실은 정제된 예제에 대한 VQLA 손실($\mathcal{L}_{VQLA}$), 섭동된 예제에 대한 VQLA 손실($\mathcal{L}'_{VQLA}$), 그리고 대조 손실($\mathcal{L}_{CTR}$)의 합입니다:
        $$\mathcal{L} = \mathcal{L}_{VQLA}(x; G) + \mathcal{L}_{VQLA}(x+r; G) + \mathcal{L}_{CTR}(x; r)$$

## 📊 Results

* **Surgical-VQLA [3] 대비 성능 향상**: Surgical-VQLA++는 EndoVis-18-VQLA 및 EndoVis-17-VQLA 데이터셋 모두에서 Accuracy와 mIoU 지표에서 1.06%에서 11.59%까지 상당한 성능 개선을 보였습니다.
* **SOTA 모델 대비 우수성**: EndoVis-18-VQLA 데이터셋에서 Surgical-VQLA++는 전체 성능에서 Accuracy 1.12%, mIoU 1.55% 향상으로 모든 기준 모델을 능가했습니다. 특히, 견고성 테스트에서는 Accuracy 1.64%, mIoU 1.70% 더 뛰어난 성능을 보였습니다.
* **모듈별 기여도**: co-attention 레이어, gated fusion 모듈, multimodal collaborated calibration, global contextual calibration, C$^2$G-ViL 임베딩 중 어느 하나라도 제거하면 모델 성능이 크게 저하되어, 제안된 모든 모듈이 최상의 결과 달성에 기여함을 입증했습니다.
* **다양한 질문 유형에 대한 성능**: 조직 인식(tissue recognition) 질문에서는 모든 모델이 만점을 받았으며, 동작 상태 인식(action state recognition) 및 도구 위치(instrument location) 질문에서 본 모델이 최고의 성능을 달성했습니다. 도구 식별(instrument identification)에서는 답변 성능이 상대적으로 낮았지만, 로컬라이제이션 성능은 높게 유지되었습니다.
* **시각 특징 추출기 영향**: ResNet18을 사용한 end-to-end 아키텍처는 Faster RCNN 기반 방식보다 추론 속도가 현저히 빠르고 정량적 성능도 우수했습니다.
* **손실 함수 조합 분석**: Focal Loss와 GIoU Loss, 그리고 불확실성 손실의 조합이 최적의 성능을 제공했습니다. GIoU Loss가 다른 IoU 손실(DIoU, CIoU)보다 멀티태스크 학습 프레임워크에 가장 적합했습니다.
* **융합 전략 분석**: 제안된 `C$^2$G-ViL` 임베딩 전략이 다른 정교한 멀티모달 융합 방법론들(Concatenation, JCA, MMHCA, MAT, GMU, Self-Attention, Guided-Attention, Co-Attention)보다 우수했습니다. 특히, 텍스트 정보가 시각 특징 임베딩을 가이드하는 방식(T2V)이 VQLA 태스크에서 뛰어난 성능을 보였습니다.
* **적대적 섭동 가중치 최적화**: 그리드 서치를 통해 섭동 가중치 $\alpha = 1, \beta = 0.5$가 최적의 성능을 보임을 확인했습니다. 적대적 대조 학습 전략을 제외하면 성능이 크게 저하되었습니다.
* **이미지 손상에 대한 견고성**: 노이즈, 블러, 가려짐, 디지털 손상 등 19가지 유형의 이미지 손상에 대해 강건성을 평가했습니다. Surgical-VQLA++는 손상 정도가 증가함에 따라 성능 저하가 일관되게 나타났지만, 기존 Surgical-VQLA 및 CAT-ViL 방식보다 지속적으로 높은 Accuracy와 mIoU를 달성하여 우수한 견고성을 입증했습니다.

## 🧠 Insights & Discussion

본 연구는 `C$^2$G-ViL` 모듈과 적대적 대조 학습을 통해 수술 VQLA의 성능과 견고성을 크게 향상시켰습니다. 멀티모달 표현의 정렬과 전역 컨텍스트 특징 보정은 특징 융합 시 정보 교환을 강화하고, 최적의 융합 가중치 및 대조 학습 전략은 미묘한 패턴을 인식하고 포착하는 데 도움이 됩니다.

**한계점**: 특정 VQLA 태스크 인스턴스에서 F-Score 결과가 다른 정교한 접근 방식보다 낮게 나타났습니다. 이는 특정 시나리오에서 모델의 성능이 저하될 수 있음을 시사합니다. 향후 연구에서는 모든 관련 지표를 균형 있게 고려하면서 정밀도(precision)와 재현율(recall)을 개선하기 위한 최적의 정보 융합 전략을 탐색할 필요가 있습니다. 또한, 도메인 이동(domain shift) 상황에서 VQLA 태스크의 정확도 요구 사항을 완전히 충족하지 못할 수 있다는 점을 인정하며, 도메인 적응 및 연속 학습 전략, 멀티모달 융합 특징에 대한 기울기 섭동 기법 탐색 등이 미래 연구 방향이 될 수 있습니다.

**MLLM과의 비교**: 현재의 멀티모달 대규모 언어 모델(MLLM)은 인간과 유사한 상호작용과 복잡한 응답을 생성할 수 있지만, 느린 응답 속도, 높은 GPU 요구 사항, 도메인 특화 문제에 대한 미세 조정 필요성, 환각 및 반복적인 출력 문제가 있습니다. 반면, Surgical-VQLA++는 특정 수술 이미지에 대한 정확하고 문맥 인식적인 응답을 제공하여 진단 및 치료 결정에 중요하게 기여합니다. 특히, 실시간 의사 결정이 중요한 수술 환경에서 즉각적인 시각 지원을 제공하며, 적대적 대조 학습을 통해 이미지 손상에 대한 견고성을 확보합니다. 수술 보고서나 캡션 생성은 일반적인 텍스트 설명을 제공하는 반면, VQLA 모델은 주어진 질문에 대한 특정 답변과 해당 로컬라이제이션 결과를 제공하여 수술 맥락에서 시각 정보 처리의 정확성과 유용성을 향상시킵니다.

## 📌 TL;DR

본 논문은 로봇 수술에서 영상 질문 답변 및 로컬라이제이션(VQLA)의 견고성과 성능을 향상시키기 위해 **Surgical-VQLA++**를 제안합니다. 이미지 손상 및 도메인 변화에 강건하게 대응하기 위해, 멀티모달 정보를 효과적으로 정렬하고 융합하는 **C$^2$G-ViL 임베딩 모듈**과 **적대적 대조 학습 전략**을 통합했습니다. 확장된 EndoVis 데이터셋을 이용한 실험을 통해 Surgical-VQLA++가 기존 모델 대비 질문 답변 정확도와 로컬라이제이션 성능, 그리고 실제 환경의 이미지 손상에 대한 견고성 측면에서 우수함을 입증했습니다.
