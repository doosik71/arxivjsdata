# Surgical-VQLA: Transformer with Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery

Long Bai, Mobarakol Islam, Lalithkumar Seenivasan and Hongliang Ren

## 🧩 Problem to Solve

수술 영상 분석 분야에서 주니어 레지던트들은 수술 절차에 대한 질문에 답하기 위해 여전히 전문가에게 크게 의존합니다. 하지만 전문가들은 임상 및 학술 업무로 인해 시간적 여유가 부족합니다. 기존의 시각 질의 응답(VQA) 방법들은 수술 장면 이해를 돕지만, 다음과 같은 한계를 가집니다:

* **객체 검출 모델의 희소성**: 작은 데이터셋과 바운딩 박스 주석 부족으로 인해 수술 객체 검출 모델이 부족합니다.
* **이종 모달리티 융합의 미흡**: 텍스트와 이미지 같은 이종 모달리티의 특징 융합 전략이 단순합니다.
* **국소화된 답변의 부재**: 복잡한 수술 시나리오에서 중요한 '어디에(where)?' 해당하는 국소화된(localized) 답변 기능이 없습니다.

이 연구는 이러한 문제를 해결하여 '무엇(what)?'과 '어디(where)?'에 대한 답변을 제공함으로써 궁극적으로 '왜(why)?'에 대한 추론을 돕는 것을 목표로 합니다.

## ✨ Key Contributions

* 주어진 질문과 수술 장면을 기반으로 국소화된 답변을 예측하는 **Surgical Visual Question Localized-Answering (Surgical-VQLA)** 모델을 제안합니다.
* 새로운 **Gated Vision-Language Embedding (GVLE)** 기술을 사용하여 이종 특징(시각 및 텍스트)을 효과적으로 융합하는 검출기 없는(detection-free) **GVLE-LViT** 모델을 제안합니다.
* VQLA 모델의 예측 및 국소화 성능을 향상시키기 위해 교차 엔트로피($L_{\text{CE}}$) 손실 및 $L_1$ 손실과 함께 **GIoU (Generalized Intersection over Union)** 손실을 통합합니다.
* Surgical-VQLA가 수술 상호작용과 관련된 맥락도 국소화할 수 있음을 입증합니다.
* 검출기 없는 접근 방식을 통해 계산 비용이 많이 드는 검출 모듈을 피하고, 실시간 수술 질의 응답 시스템의 **엔드-투-엔드(end-to-end) 적용**을 가능하게 합니다.
* GVLE가 시각 및 단어 임베딩의 이종 모달리티를 효과적으로 융합하며 기존 접근 방식보다 우수한 성능을 보임을 확인합니다.

## 📎 Related Works

* **기존 VQA 방법**: 대부분 객체 검출 모델과 영역 기반 특징 추출기에 의존하며, 엔드-투-엔드가 아니며, 이종 특징(시각 및 텍스트) 융합이 단순합니다.
* **MedFuseNet [1] 및 Surgical-VQA [2]**: 의료 VQA 분야에서 '무엇(what)?'을 묻는 질문에 답할 가능성을 보여주었지만, '왜(why)?' 또는 '어디(where)?'에 대한 답변은 제공하지 못합니다.
* **VisualBERT [9] 및 VisualBERT ResMLP [2]**: BERT 기반의 언어 모델에 시각적 특징을 결합한 Transformer 인코더 모델로, 본 연구의 비교 대상으로 사용됩니다. 시각 및 단어 임베딩을 단순 연결(concatenation) 방식으로 융합합니다.
* **AFF (Attentional Feature Fusion) 및 iAFF (iterative Attentional Feature Fusion) [16]**: 이종 특징 융합을 위한 어텐션 기반 기법으로, 본 연구의 GVLE와 비교됩니다.

## 🛠️ Methodology

본 연구는 효율적인 임베딩을 통해 Surgical-VQLA를 수행하기 위한 **GVLE-LViT** (Gated Vision-Language Embedding Language-Vision Transformer)를 제안합니다.

* **GVLE-LViT 구성**:
  * **시각 특징 추출기 (Visual Feature Extractor)**: ImageNet [24]으로 사전 학습된 ResNet18 [23]을 사용하여 이미지 전체에서 특징을 추출합니다. 이는 객체 검출 모델을 사용하지 않아 컴퓨팅 비용을 절감합니다.
  * **토크나이저 (Tokenizer)**: 수술 관련 데이터셋으로 사용자 정의 학습된 토크나이저를 사용하여 단어 임베딩을 생성합니다.
  * **Gated Vision-Language Embedding (GVLE)**: 이종 모달리티 특징 융합을 위한 핵심 모듈입니다.
    * VisualBERT [9] 및 VisualBERT ResMLP [2]의 단순 연결 방식을 대체합니다.
    * `tanh` 활성화 함수를 통해 각 모달리티의 내부 표현을 인코딩하고, 게이트 노드 $\alpha$를 사용하여 시각 및 단어 임베딩 정보의 유용성을 제어합니다.
    * 융합 방정식은 다음과 같습니다:
            $$ \omega = \alpha(\theta_{\omega} \cdot [f\|e]) $$
            $$ \Upsilon = \omega \ast \tanh (\theta_{f} \cdot f) + (1-\omega) \ast \tanh (\theta_{e} \cdot e) $$
            여기서 $f$는 시각 임베딩, $e$는 단어 임베딩, $\theta$는 학습 가능한 파라미터, $[f\|e]$는 연결 연산, $\Upsilon$는 GVLE 모듈의 최종 출력입니다.
  * **Vision Transformer (ViT)**: GVLE의 출력 임베딩은 사전 학습된 ViT [21] Transformer 인코더를 통과합니다.
  * **예측 헤드 (Prediction Head)**:
    * **분류 헤드 (Classification Head)**: ViT 블록의 출력은 Softmax가 적용된 선형 예측 레이어를 통해 답변 분류를 수행합니다.
    * **국소화 헤드 (Localization Head)**: ReLU 활성화 함수를 가진 3-계층 퍼셉트론(FFN)과 선형 투사 레이어로 구성되며, 바운딩 박스의 정규화된 좌표(높이, 너비, 중심 좌표)를 예측합니다.
* **손실 함수 (Loss Function)**: 분류 손실과 검출 손실을 함께 사용하여 공동 학습합니다.
    $$ L = L_{\text{CE}} + (L_{\text{GIoU}} + L_1) $$
  * $L_{\text{CE}}$: 분류를 위한 교차 엔트로피 손실.
  * $L_{\text{GIoU}}$ [17]: 바운딩 박스 회귀를 위한 GIoU 손실. 겹치는 영역과 겹치지 않는 영역 모두에 초점을 맞춥니다.
  * $L_1$: 바운딩 박스 회귀를 위한 $L_1$ 손실.

## 📊 Results

* **정량적 결과**: EndoVis-18-VQLA 및 EndoVis-17-VQLA 데이터셋에서 VisualBERT [9] 및 VisualBERT ResMLP [2]와 같은 최신 Transformer 기반 모델보다 우수한 성능을 달성했습니다 (Table I).
  * GVLE-LViT는 두 데이터셋 모두에서 분류 정확도(Acc), F-점수(F-Score), mIoU에서 가장 높은 성능을 보였습니다.
  * 객체 검출 모델 없이 이미지 전체에서 특징을 추출하는 방식이 기존 객체 검출 기반 방식보다 일관되게 우수한 성능을 보였으며, 이는 모델의 전역 장면 이해 능력 덕분입니다.
  * 검출기 없는 모델은 처리 속도를 8배 이상 향상시켜 150.6 FPS를 달성하여 실시간 적용에 적합함을 입증했습니다.
* **정성적 결과**: Fig. 3에서 볼 수 있듯이, 본 모델은 답변 정확도와 국소화(Ground-truth 바운딩 박스에 근접) 모두에서 기준 모델보다 뛰어난 성능을 보였습니다.
* **K-fold 교차 검증**: 3가지 K-fold 설정에서도 GVLE-LViT 모델이 기준 Transformer 기반 모델보다 전반적으로 우수한 성능을 보였습니다 (Table II).
* **어블레이션 스터디 (Ablation Studies)**:
  * **손실 함수 조합**: $L_{\text{CE}}$ 및 $L_1$ 손실에 GIoU [17] 손실을 추가할 때 답변 예측과 국소화 성능이 모두 크게 향상됨을 확인했습니다 (Table III).
  * **융합 기법 비교**: GVLE 시각-언어 특징 융합 기법이 ConCAT [9], AFF [16], iAFF [16] 등 다른 특징 융합 기법보다 우수함을 입증했습니다 (Table IV).

## 🧠 Insights & Discussion

* 제안된 Surgical-VQLA 모델은 수술 교육에서 중요한 보조 도구가 될 수 있습니다. '무엇?'과 '어디?'에 대한 답변을 제공함으로써 주니어 의사들이 '왜?'에 대한 추론을 더 쉽게 할 수 있도록 돕습니다.
* 답변의 국소화 정보는 새로운 데이터에 대한 예측 신뢰도를 평가하는 데 활용될 수 있습니다. 만약 국소화된 영역이 대상 도구 또는 조직에서 크게 벗어난다면, 예측이 잘못되었거나 입력 데이터가 분포 외(out-of-distribution) 데이터일 가능성을 추론할 수 있습니다.
* **한계 및 향후 연구**:
  * 국소화 정보를 활용하여 예측 신뢰도를 예측하는 것이 가능한 미래 연구 방향입니다.
  * 더 복잡한 데이터셋과 도전적인 질의응답 쌍을 사용하여 Surgical-VQLA 시스템의 잠재력을 더욱 증진시킬 수 있습니다.
  * 의료 진단을 위한 새로운 응용 가능성을 열 수 있습니다.

## 📌 TL;DR

수술 훈련에서 주니어 의사들의 질문에 대한 전문가 의존도를 줄이고자, 본 논문은 질문에 대한 답변과 함께 관련 수술 영역을 시각적으로 국소화하는 **Surgical-VQLA** 모델을 제안합니다. 특히, 새로운 **Gated Vision-Language Embedding (GVLE)**을 사용하여 이종 시각-언어 특징을 효과적으로 융합하고, **GIoU 손실**을 통합하여 예측 및 국소화 성능을 향상시킨 **GVLE-LViT**를 개발했습니다. 이 모델은 기존 최신 VQA 모델보다 뛰어난 성능을 보이며, 검출기 없는(detection-free) 설계로 **실시간 처리 능력**을 갖추어 수술 교육의 중요한 보조 도구가 될 잠재력을 보여주었습니다.
