from aidose.ctgov.structures import Study
from aidose.ctgov.utils_pdf import extract_text_from_pdf

from typing import List, Any
import requests
import os


# =========================
# Intervention accessors
# =========================

def get_protocol_interventions(study: Study) -> List[Any]:
    ps = study.protocolSection
    if not ps or not ps.armsInterventionsModule:
        return []
    return ps.armsInterventionsModule.interventions or []


def get_protocol_arm_groups(study: Study) -> List[Any]:
    ps = study.protocolSection
    if not ps or not ps.armsInterventionsModule:
        return []
    return ps.armsInterventionsModule.armGroups or []


# =========================
# Document helpers
# =========================

def _has_doc_flag(study: Study, flag_name: str) -> bool:
    ds = study.documentSection
    if not ds or not ds.largeDocumentModule:
        return False
    large_docs = ds.largeDocumentModule.largeDocs or []
    for doc in large_docs:
        val = getattr(doc, flag_name, None)
        if isinstance(val, bool) and val:
            return True
    return False


def has_protocol(study: Study) -> bool: return _has_doc_flag(study, "hasProtocol")


def has_sap(study: Study) -> bool:       return _has_doc_flag(study, "hasSap")


def has_icf(study: Study) -> bool:       return _has_doc_flag(study, "hasIcf")


def get_large_protocols_pdf_links(study: Study, check_link_status: bool = False) -> List[str] | None:
    if not has_protocol(study):
        return None
    large_docs = study.documentSection.largeDocumentModule.largeDocs
    if not large_docs:
        return None
    links: List[str] = []
    nctid = study.protocolSection.identificationModule.nctId
    subfolder = nctid[-2:]  # Last two characters of NctId
    for doc in large_docs:
        filename = doc.filename
        if isinstance(filename, str) and filename.endswith(".pdf"):
            link = "https://cdn.clinicaltrials.gov/large-docs/{}/{}/{}".format(subfolder, nctid, filename)
            if check_link_status:
                try:
                    response = requests.head(link, timeout=5)
                    if response.status_code != 200:
                        raise RuntimeError(f"URL not found or not accessible: {link} (status {response.status_code})")

                except requests.RequestException as e:
                    raise RuntimeError(f"Error checking URL {link}: {e}")

            links.append(link)

    return links if links else None


def get_protocol_pdfs_saved_dir_for_nctid(nctid: str, extensions_dir: str) -> str | None:
    parent_identifier = nctid[-2:]
    return os.path.join(extensions_dir, "protocol-pdfs", parent_identifier, nctid)


def concatenate_pdf_texts_for_nctid(trial_nctid: str, extensions_dir: str) -> str | None:
    pdf_dir = get_protocol_pdfs_saved_dir_for_nctid(trial_nctid, extensions_dir)
    if not os.path.exists(pdf_dir):
        return None
    all_texts = []
    for pdf_name in [_file for _file in os.listdir(pdf_dir) if _file.endswith(".pdf")]:
        pdf_path = os.path.join(pdf_dir, pdf_name)
        text = extract_text_from_pdf(pdf_path)
        all_texts.append("{}: \n{}".format(pdf_name, text))
    return "\n".join(all_texts) if all_texts else None
