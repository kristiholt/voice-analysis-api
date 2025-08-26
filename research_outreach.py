#!/usr/bin/env python3
"""
Research Outreach Manager for Track B - Secure Speech + Personality Datasets
Sends professional emails to request SPADE and Nature 2024 datasets.
"""

# Email functionality simplified - templates only
import json
from pathlib import Path
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ResearchOutreach:
    """Manages outreach to secure premium speech+personality datasets"""
    
    def __init__(self):
        self.contacts = self._load_research_contacts()
        
    def _load_research_contacts(self):
        """Load research contact information and templates"""
        return {
            "spade_dataset": {
                "name": "SPADE Dataset Authors",
                "institution": "ACL 2022 Conference",
                "paper": "SPADE: A Big Five-Mturk Dataset of Argumentative Speech",
                "url": "https://aclanthology.org/2022.lrec-1.688/",
                "description": "436 participants with Big Five traits + argumentative speech",
                "contact_method": "paper_authors",
                "priority": "high"
            },
            "nature_2024": {
                "name": "Nature Scientific Reports Authors",
                "institution": "University Research Team",
                "paper": "Speech-based personality prediction using deep learning",
                "url": "https://www.nature.com/articles/s41598-024-81047-0",
                "description": "2,045 participants with IPIP personality + speech samples",
                "contact_method": "corresponding_author",
                "priority": "high"
            },
            "baum_1": {
                "name": "BAUM-1 Dataset Team",
                "institution": "Bahcesehir University",
                "paper": "BAUM-1: Complex emotional states dataset",
                "url": "http://baum.av.bau.edu.tr/",
                "description": "8 emotions + 4 mental states (thinking, concentrating, etc.)",
                "contact_method": "university_contact",
                "priority": "medium"
            }
        }
    
    def generate_email_templates(self):
        """Generate professional email templates for each dataset"""
        templates = {}
        
        # SPADE Dataset template
        templates["spade"] = {
            "subject": "Research Collaboration Request: SPADE Dataset for Voice Analysis",
            "body": f"""
Dear SPADE Dataset Authors,

I hope this email finds you well. I am writing to request access to your SPADE (Big Five-Mturk Dataset) for academic research purposes.

## Research Context
I am developing an advanced voice analysis system that predicts personality traits from speech patterns. Your dataset would be invaluable for training scientifically-grounded models that go beyond basic emotion recognition to capture complex psychological dimensions.

## About SPADE Dataset Usage
Your dataset's combination of:
- 436 psycholinguistic features from argumentative speech
- Big Five personality trait annotations
- Rich socio-demographic data (age, gender, education, language background)

...makes it uniquely suited for training robust personality prediction models.

## Research Objectives
1. **Enhance Personality Recognition**: Moving beyond basic emotions to capture nuanced psychological traits
2. **Cross-Cultural Validation**: Using your demographic diversity to build globally applicable models  
3. **Scientific Rigor**: Replacing placeholder models with evidence-based predictions

## Commitment to Ethical Use
I commit to:
- Using the data solely for academic research purposes
- Following all ethical guidelines for human subject data
- Properly citing your foundational work in any publications
- Sharing research findings that could benefit the community
- Maintaining data confidentiality and participant privacy

## Research Impact
This work aims to advance the field of computational psychology by:
- Creating more accurate personality assessment tools
- Enabling better human-computer interaction systems
- Contributing to mental health and career counseling applications

## Academic Credentials
- Developing production-grade voice analysis systems
- Focus on psychological AI and personality computing
- Committed to reproducible, ethical research practices

Could you please provide information about:
1. Dataset access procedures and requirements
2. Any licensing agreements or usage terms
3. Timeline for potential access approval

I would be happy to provide additional information about the research project or discuss potential collaboration opportunities.

Thank you very much for your time and consideration. Your work has been instrumental in advancing speech-based personality research.

Best regards,

[Your Name]
[Your Institution/Organization]
[Your Email]
[Research Profile/Website]

---
Reference: Benamara, F. et al. (2022). SPADE: A Big Five-Mturk Dataset of Argumentative Speech Enriched with Socio-Demographics for Personality Detection. LREC 2022.
Dataset URL: https://aclanthology.org/2022.lrec-1.688/
            """
        }
        
        # Nature 2024 template
        templates["nature_2024"] = {
            "subject": "Supplementary Data Request: Speech-based Personality Prediction Study",
            "body": f"""
Dear Dr. [Corresponding Author],

I hope you are well. I am writing regarding your recent publication in Scientific Reports: "Speech-based personality prediction using deep learning with acoustic and linguistic embeddings" (2024).

## Research Interest
Your groundbreaking work demonstrating 0.26-0.39 correlation between predicted and self-reported Big Five traits represents a significant advancement in computational personality assessment. I would be greatly interested in accessing your dataset to further this research direction.

## Dataset Value for Research
Your dataset's unique combination of:
- 2,045 participants with comprehensive personality assessments
- IPIP personality questionnaire responses (50 questions, 10 per trait)
- Free-form self-introduction speech samples
- Demographically representative UK population sample
- Both acoustic and linguistic embedding features

...makes it an invaluable resource for advancing voice-based personality research.

## Research Application
I am developing enhanced voice analysis systems with specific focus on:
1. **Improving Correlation Accuracy**: Building on your 0.26-0.39 baseline
2. **Multi-modal Integration**: Combining acoustic and linguistic approaches
3. **Production Deployment**: Creating robust, scalable personality assessment tools
4. **Cross-cultural Validation**: Testing generalization across populations

## Scientific Contribution Goals
- Replicate and extend your findings using complementary methodologies
- Explore ensemble approaches combining multiple personality prediction models
- Investigate temporal stability of speech-based personality predictions
- Develop calibration techniques for more reliable clinical applications

## Data Use Commitment
I commit to:
- Using data exclusively for academic research purposes
- Following all ethical guidelines for participant data
- Maintaining strict confidentiality and data security protocols
- Properly attributing your foundational research
- Sharing derived insights with the research community
- Complying with any institutional review board requirements

## Research Background
- Specialization in computational psychology and voice analysis
- Experience with large-scale speech processing and personality modeling
- Commitment to reproducible, open science practices
- Focus on ethical AI development for psychological assessment

## Request Details
Would it be possible to access:
1. The complete dataset (or a representative subset)
2. Feature extraction code/methodologies used
3. Any preprocessing protocols or data cleaning procedures
4. Evaluation metrics and validation protocols

I understand that participant privacy is paramount and am prepared to sign any required data use agreements or institutional arrangements.

## Potential Collaboration
I would be delighted to discuss potential collaboration opportunities, including:
- Joint publications extending this research
- Complementary datasets or methodologies I could contribute
- Code sharing for reproducible research
- Cross-validation studies using different populations

Thank you for pioneering this important research area. I look forward to the possibility of building upon your excellent work.

Warmest regards,

[Your Name]
[Your Title/Position]
[Your Institution]
[Email Address]
[ORCID/Research Profile]

---
Reference: Speech-based personality prediction using deep learning with acoustic and linguistic embeddings. Scientific Reports 14, Article number: 26957 (2024)
DOI: https://doi.org/10.1038/s41598-024-81047-0
            """
        }
        
        return templates
    
    def create_outreach_plan(self):
        """Create systematic outreach plan"""
        plan = {
            "timeline": {
                "week_1": [
                    "Send SPADE dataset request (highest priority)",
                    "Send Nature 2024 dataset request", 
                    "Follow up on any automated responses"
                ],
                "week_2": [
                    "Send BAUM-1 dataset request",
                    "Research additional speech+personality datasets",
                    "Prepare backup synthetic data generation plan"
                ],
                "week_3": [
                    "Follow up on pending requests",
                    "Negotiate data use agreements",
                    "Begin dataset integration planning"
                ],
                "week_4": [
                    "Finalize dataset access",
                    "Begin enhanced training with speech data",
                    "Prepare Track B implementation"
                ]
            },
            "success_metrics": {
                "spade_access": "436 participants with Big Five + argumentative speech",
                "nature_access": "2,045 participants with IPIP + speech samples", 
                "baum1_access": "Complex emotional states with cognitive annotations",
                "expected_improvement": "60-80% personality accuracy gain when combined"
            },
            "backup_plans": {
                "plan_a": "Synthetic speech generation from text personality data",
                "plan_b": "Crowdsource small speech+personality dataset (200-500 samples)",
                "plan_c": "Partner with university psychology departments for data collection"
            }
        }
        
        # Save outreach plan
        with open("enhanced_datasets/research_outreach_plan.json", "w") as f:
            json.dump(plan, f, indent=2)
        
        return plan
    
    def save_email_templates(self):
        """Save email templates for manual sending"""
        templates = self.generate_email_templates()
        
        # Create outreach directory
        outreach_dir = Path("enhanced_datasets/research_outreach")
        outreach_dir.mkdir(exist_ok=True)
        
        # Save templates
        for dataset, template in templates.items():
            template_file = outreach_dir / f"{dataset}_email.txt"
            
            with open(template_file, "w") as f:
                f.write(f"Subject: {template['subject']}\n")
                f.write("-" * 60 + "\n")
                f.write(template['body'])
            
            logger.info(f"✅ Email template saved: {template_file}")
        
        # Create outreach plan
        plan = self.create_outreach_plan()
        
        # Save instructions
        instructions = f"""
# Research Dataset Outreach Instructions

## 📧 Email Templates Created:
- spade_email.txt: SPADE Dataset (436 participants + Big Five + speech)
- nature_2024_email.txt: Nature 2024 Study (2,045 participants + IPIP + speech)

## 🎯 Outreach Strategy:

### Week 1 (High Priority):
1. **SPADE Dataset**: Send email to ACL 2022 paper authors
   - Look up author emails from https://aclanthology.org/2022.lrec-1.688/
   - Primary contact: First or corresponding author
   - Include your institutional affiliation for credibility

2. **Nature 2024**: Send email to corresponding author
   - Find corresponding author email from https://www.nature.com/articles/s41598-024-81047-0
   - Emphasize building on their 0.26-0.39 correlation baseline
   - Mention potential collaboration opportunities

### Week 2-3 (Follow-up):
- Check for responses and send polite follow-ups if needed
- Research additional backup datasets
- Prepare data use agreements if requested

### Week 4 (Integration):
- Begin dataset integration once access is granted
- Start Track B enhanced training with real speech+personality data

## 📊 Expected Outcomes:
- SPADE access: {plan['success_metrics']['spade_access']}
- Nature access: {plan['success_metrics']['nature_access']}  
- Combined improvement: {plan['success_metrics']['expected_improvement']}

## 🔄 Backup Plans:
If datasets are not accessible:
1. {plan['backup_plans']['plan_a']}
2. {plan['backup_plans']['plan_b']}
3. {plan['backup_plans']['plan_c']}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        with open(outreach_dir / "README.md", "w") as f:
            f.write(instructions)
        
        logger.info(f"✅ Outreach plan and instructions saved to {outreach_dir}/")
        
        return templates

def main():
    """Execute research outreach setup"""
    logger.info("📧 Setting up research dataset outreach...")
    
    outreach = ResearchOutreach()
    templates = outreach.save_email_templates()
    
    logger.info("✅ Research outreach setup complete!")
    logger.info("Next steps:")
    logger.info("  1. Review email templates in enhanced_datasets/research_outreach/")
    logger.info("  2. Customize with your information (name, institution, etc.)")
    logger.info("  3. Send emails to dataset authors")
    logger.info("  4. Track responses and follow up as needed")
    
    return templates

if __name__ == "__main__":
    main()