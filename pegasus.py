import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def summarize_with_pegasus(text: str, model_name: str = "google/pegasus-xsum", max_len: int = 128) -> str:
    """
    Summarize text using Pegasus (via AutoTokenizer and AutoModelForSeq2SeqLM).
    """
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    # Encode with truncation and padding to avoid index errors
    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="longest"
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



if __name__ == "__main__":
    
    text = """
     Prison Link Cymru had referrals in and said some were living rough for up to a year before finding suitable accommodation Workers at the charity claim investment in housing would be cheaper than jailing homeless repeat offenders The Welsh Government said more people than ever were getting help to address housing problems Changes to the Housing Act in Wales introduced in removed the right for prison leavers to be given priority for accommodation Prison Link Cymru which helps people find accommodation after their release said things were generally good for women because issues such as children or domestic violence were now considered However the same could not be said for men the charity said because issues which often affect them such as post traumatic stress disorder or drug dependency were often viewed as less of a priority Andrew Stevens who works in Welsh prisons trying to secure housing for prison leavers said the need for accommodation was chronic There a desperate need for it finding suitable accommodation for those leaving prison there is just a lack of it everywhere he said It could take six months to a year without a lot of help they could be on the streets for six months When you think of the consequences of either being on the street especially with the cold weather at the moment or you may have a roof over your head sometimes there is only one choice Mr Stevens believes building more flats could help ease the problem The average price is a hundred pounds a week to keep someone in a rented flat prison is a lot more than that so I would imagine it would save the public purse quite a few pounds he said Official figures show properties were built in the year to March of an overall total of new properties in Wales Marc who has been in and out of prison for the past years for burglary offences said he struggled to find accommodation each time he was released He said he would ask himself Where am I going to stay Where am I going to live Have I got somewhere where I can see my daughter You put out among the same sort of people doing the same sort of thing and it difficult it difficult to get away from it It like every man for himself there nothing Marc has now found stable accommodation with homeless charity Emmaus and said it had been life changing You feel safe you got hot food you got company of people in similar situations to yourself but all dealing with different issues It a constructive helpful atmosphere he said Tom Clarke chief executive of Emmaus South Wales agreed there was not enough support available We do still see people homeless on the streets so clearly they have got accommodation and have got provision he said I think the key is connecting people with the services they need I do delude myself that Emmaus can offer a one size fits all for everyone we ca But there must be other opportunities and given suitable encouragement I believe that can and should happen A Welsh Government spokesman said the national pathway for homeless services to children young people and adults in the secure estate had prevented many people from losing their home whilst serving their prison sentence It added there were already significant demands for flats across the public and private sector and it was providing new affordable homes in the next five years
    """    
    print("Pegasus Summary:", summarize_with_pegasus(text))
