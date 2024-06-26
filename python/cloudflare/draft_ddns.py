import requests
import cloudflare

ENV = {
    'CLOUDFLARE_EMAIL': 'xxxx',
    'CLOUDFLARE_API_KEY': 'yyyy',
    'DOMAIN_NAME': 'zzz',
    'ZONE_ID': None, # will be set later
    'SUBDOMAIN': 'ssh-hk3060',
}

def get_public_ip(ipv6:bool=False):
    # https://www.ipify.org/
    # https://stackoverflow.com/a/3097641/7290857
    url = 'https://api64.ipify.org' if ipv6 else 'https://api.ipify.org'
    ret = requests.get(url, params={'format':'json'}).json()['ip']
    # ret = requests.get(url).text
    return ret


if __name__ == "__main__":
    public_ip = get_public_ip()

    client = cloudflare.Cloudflare(api_email=ENV["CLOUDFLARE_EMAIL"], api_key=ENV["CLOUDFLARE_API_KEY"])
    tmp0 = [x for x in client.zones.list() if x.name==ENV['DOMAIN_NAME']]
    assert len(tmp0)==1, 'Domain not found'
    ENV['ZONE_ID'] = tmp0[0].id

    for record in client.dns.records.list(zone_id=ENV['ZONE_ID']):
        if record.name==ENV['SUBDOMAIN']+'.'+ENV['DOMAIN_NAME']: #find exist
            if record.content!=public_ip:
                client.dns.records.update(dns_record_id=record.id, zone_id=ENV['ZONE_ID'], content=public_ip, name=f'{ENV["SUBDOMAIN"]}.{ENV["DOMAIN_NAME"]}', type='A', ttl=120, proxied=False)
            break
    else: #not exist
        # Time to live (ttl), 120 seconds is the minimum value
        client.dns.records.create(zone_id=ENV['ZONE_ID'], name=f'{ENV["SUBDOMAIN"]}.{ENV["DOMAIN_NAME"]}', type='A', content=public_ip, ttl=120, proxied=False)
