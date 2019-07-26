import argparse
import torch
from tqdm import tqdm
import Seqlib.data_loader as module_data
import Seqlib.model.loss as module_loss
# import model.metric as module_metric
import Seqlib.model as module_arch
from Seqlib.parse_config import ConfigParser


def decode_word(decoder_output, dataset):
    output_lang = dataset.output_lang
    topv, topi = decoder_output.topk(1, dim=1)
    topi.squeeze_(dim=1)
    decoded_words = []
    for i in range(topi.size(0)):
        decoded_words.append(output_lang.index2word[topi[i].item()])
    return ' '.join(decoded_words)

def evaluate(decoder_output, target, dataset):
    decoded_words = decode_word(decoder_output, dataset)
    output_lang = dataset.output_lang
    target_words = []
    for i in range(target.size(0)):
        target_words.append(output_lang.index2word[target[i].item()])
    return ' '.join(target_words), decoded_words

def batch_evaluate(decoder_outputs, targets, dataset):
    for i in range(len(decoder_outputs)):
        print(evaluate(decoder_outputs[i], targets[i], dataset))


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        lang1='eng',
        lang2='deu',
        batch_size=50,
        mode="train",
        shuffle=False,
        validation_split=0.0,
        num_workers=4
    )

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)
    model.teaching = False

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output, _ = model(data, target)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size

            batch_evaluate(output, target, data_loader.dataset)
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Seqlib')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)
